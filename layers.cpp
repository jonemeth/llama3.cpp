#include <algorithm>
#include <cmath>
#include <numeric>

#include "layers.h"

namespace detail
{

inline void matmulFloat(FloatTensor &xout, FloatTensor const &x, FloatTensor const &w)
{
    size_t i;
#pragma omp parallel for private(i)
    for (i = 0; i < xout.size(); i++)
        xout[i] = std::inner_product(x.begin(), x.end(), w.begin() + i * x.size(), 0.0f);
}

inline void matmulQuantized(FloatTensor &xout, QuantizedTensor const &x, QuantizedTensor const &w)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    int groupSize = x.groupSize;

    int n = x.q.size();
    int d = xout.size();
    int i = 0;

#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        int in = i * n;

        // do the matmul in groups of GS
        for (int j = 0; j <= n - groupSize; j += groupSize)
        {
            float ival = static_cast<float>(
                std::inner_product(x.q.data() + j, x.q.data() + j + groupSize, w.q.data() + in + j,
                                   static_cast<int32_t>(0), std::plus<int32_t>(), [](int8_t x, int8_t w)
                                   { return static_cast<int32_t>(x) * static_cast<int32_t>(w); }));

            val += (ival)*w.s[(in + j) / groupSize] * x.s[j / groupSize];
            ival = 0;
        }

        xout[i] = val;
    }
}

inline void applyRotaryEmbedding(FloatTensor &q, FloatTensor &k, size_t pos, size_t nHeads, size_t headSize,
                                 size_t n_kv_heads)
{
    // RoPE relative positional encoding: complex-valued rotate q and k in
    // each head
    for (size_t i = 0; i < nHeads; i++)
    {
        for (size_t j = 0; j < headSize; j += 2)
        {
            float freq = 1.0f / powf(500000.0f, (float)j / (float)headSize);
            float val = pos * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);
            float q0 = q[i * headSize + j];
            float q1 = q[i * headSize + j + 1];
            q[i * headSize + j] = q0 * fcr - q1 * fci;
            q[i * headSize + j + 1] = q0 * fci + q1 * fcr;
            if (i < n_kv_heads)
            {
                float k0 = k[i * headSize + j];
                float k1 = k[i * headSize + j + 1];
                k[i * headSize + j] = k0 * fcr - k1 * fci;
                k[i * headSize + j + 1] = k0 * fci + k1 * fcr;
            }
        }
    }
}

inline void softmax(float *first, float *last)
{
    // find max value (for numerical stability)
    float maxVal = *std::max_element(first, last);

    // exp and sum
    float sum = 0.0f;
    for (auto it = first; it != last; ++it)
    {
        *it = std::exp(*it - maxVal);
        sum += *it;
    }

    // normalize
    for (auto it = first; it != last; ++it)
        *it /= sum;
}

} // namespace detail

RMSNorm::RMSNorm(size_t dim)
    : dim(dim),
      weight(dim)
{
}

void RMSNorm::forward(Tensor &x, Tensor &out)
{
    auto xf = x.cf().data();
    auto outf = out.f().data();
    auto wf = weight.cf().data();

    float ss = std::inner_product(xf, xf + dim, xf, 0.0f);
    ss = 1.0f / std::sqrt(1e-5f + ss / dim);

    // normalize and scale
    std::transform(xf, xf + dim, wf, outf, [ss](float x, float w) { return w * (ss * x); });
}

void RMSNorm::loadWeights(std::ifstream &inputStream) { weight.readFromFile(inputStream); }

Linear::Linear(size_t inDim, size_t outDim)
    : inDim(inDim),
      outDim(outDim),
      weight(inDim * outDim)
{
}

void Linear::forward(Tensor &x, Tensor &out)
{
    if (x.size() != inDim || out.size() != outDim)
        throw std::runtime_error("Dimension mismatch!");

    if (weight.isQuantizedValid())
    {
        auto groupSize = weight.cq().groupSize;
        detail::matmulQuantized(out.f(), x.cq(groupSize), weight.cq());
    }
    else
        detail::matmulFloat(out.f(), x.cf(), weight.cf());
}

void Linear::loadWeights(std::ifstream &inputStream) { weight.readFromFile(inputStream); }

template <typename T> void Linear::setWeights(T const &w) { weight = w; }
template void Linear::setWeights<Tensor>(Tensor const &w);
template void Linear::setWeights<FloatTensor>(FloatTensor const &w);
template void Linear::setWeights<QuantizedTensor>(QuantizedTensor const &w);

CausalAttention::CausalAttention(size_t seqLength, size_t dim, size_t nHeads, size_t nKVHeads)
    : pos(0),
      dim(dim),
      nHeads(nHeads),
      nKVHeads(nKVHeads),
      wq(dim, dim),
      wk(dim, dim * nKVHeads / nHeads),
      wv(dim, dim * nKVHeads / nHeads),
      wo(dim, dim),
      query(dim, true),
      keyCache(seqLength, Tensor((dim * nKVHeads) / nHeads, true)),
      valueCache(seqLength, Tensor((dim * nKVHeads) / nHeads, true)),
      att(nHeads, Tensor(seqLength, true)),
      xb(dim, true)
{
}

void CausalAttention::forward(Tensor &x, Tensor &out)
{
    // keyCache/valueCache are used as a ring buffer: pos is the absolute
    // token position (used for RoPE), slot is where this token is stored.
    // The first nSink slots are never overwritten: the model relies on the
    // initial tokens (BOS) as "attention sinks" and degenerates without them.
    size_t const seqLength = keyCache.size();
    size_t const nSink = std::min<size_t>(4, seqLength - 1);
    size_t const slot = pos < seqLength ? pos : nSink + (pos - nSink) % (seqLength - nSink);
    size_t const nCached = std::min(pos + 1, seqLength);

    // qkv matmuls for this position
    wq.forward(x, query);
    wk.forward(x, keyCache[slot]);
    wv.forward(x, valueCache[slot]);

    size_t headSize = dim / nHeads;
    detail::applyRotaryEmbedding(query.f(), keyCache[slot].f(), pos, nHeads, headSize, nKVHeads);

    size_t kvMul = nHeads / nKVHeads; // integer multiplier of the kv sharing in multiquery
    float const scale = 1.0f / std::sqrt(static_cast<float>(headSize));

    // fetch the raw pointers before the parallel region so that threads
    // don't concurrently modify the Tensors' validity flags
    float *qf = query.f().data();
    float *xbf = xb.f().data();

    // multihead attention. iterate over all heads
    size_t h;

#pragma omp parallel for private(h)
    for (h = 0; h < nHeads; h++)
    {
        // get the query vector for this head
        float *q = qf + h * headSize;
        float *attf = att[h].f().data();

        // iterate over all cached timesteps, including the current one.
        // The keys already carry their RoPE rotation, so slot order does not matter.
        for (size_t t = 0; t < nCached; t++)
        {
            // get the key vector for this head and at this timestep
            float const *k = keyCache[t].cf().data() + (h / kvMul) * headSize;

            // calculate the attention score as the dot product of q and k
            attf[t] = std::inner_product(q, q + headSize, k, 0.0f) * scale;
        }

        // softmax the scores to get attention weights
        detail::softmax(attf, attf + nCached);

        // weighted sum of the values, store back into xb
        float *xbh = xbf + h * headSize;

        std::fill(xbh, xbh + headSize, 0.0f);
        for (size_t t = 0; t < nCached; t++)
        {
            // get the value vector for this head and at this timestep
            float const *v = valueCache[t].cf().data() + (h / kvMul) * headSize;

            // accumulate the weighted value into xb
            for (size_t i = 0; i < headSize; i++)
                xbh[i] += attf[t] * v[i];
        }
    }

    // final matmul to get the output of the attention
    wo.forward(xb, out);

    ++pos;
}
void CausalAttention::loadWeights(std::ifstream &inputStream)
{
    wq.loadWeights(inputStream);
    wk.loadWeights(inputStream);
    wv.loadWeights(inputStream);
    wo.loadWeights(inputStream);
}

FFN::FFN(size_t dim, size_t hiddenDim)
    : dim(dim),
      hiddenDim(hiddenDim),
      w1(dim, hiddenDim),
      w2(hiddenDim, dim),
      w3(dim, hiddenDim),
      hb(hiddenDim, true),
      hb2(hiddenDim, true)
{
}

void FFN::forward(Tensor &x, Tensor &out)
{
    w1.forward(x, hb);
    w3.forward(x, hb2);

    auto hbf = hb.f().data();
    auto hb2f = hb2.f().data();

    // SwiGLU non-linearity
    for (size_t i = 0; i < hiddenDim; i++)
    {
        float val = hbf[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + std::exp(-val)));
        // elementwise multiply with w3(x)
        val *= hb2f[i];
        hbf[i] = val;
    }

    // final matmul to get the output of the ffn
    w2.forward(hb, out);
}

void FFN::loadWeights(std::ifstream &inputStream)
{
    w1.loadWeights(inputStream);
    w2.loadWeights(inputStream);
    w3.loadWeights(inputStream);
}

TransformerBlock::TransformerBlock(size_t seqLength, size_t dim, size_t nHeads, size_t nKVHeads, size_t hiddenDim)
    : attentionNorm(dim),
      attention(seqLength, dim, nHeads, nKVHeads),
      ffnNorm(dim),
      ffn(dim, hiddenDim),
      xb(dim, true),
      xb2(dim, true)
{
}

void TransformerBlock::forward(Tensor &x, Tensor &out)
{
    attentionNorm.forward(x, xb);
    attention.forward(xb, xb2);

    auto xb2f = xb2.f().data();
    auto xf = x.f().data();
    for (size_t i = 0; i < x.size(); i++)
        xb2f[i] += xf[i];

    ffnNorm.forward(xb2, xb);
    ffn.forward(xb, out);

    auto outf = out.f().data();
    for (size_t i = 0; i < x.size(); i++)
        outf[i] += xb2f[i];
}

void TransformerBlock::loadWeights(std::ifstream &inputStream)
{
    attentionNorm.loadWeights(inputStream);
    attention.loadWeights(inputStream);
    ffnNorm.loadWeights(inputStream);
    ffn.loadWeights(inputStream);
}