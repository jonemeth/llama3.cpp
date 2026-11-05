#include <algorithm>
#include <cmath>
#include <numeric>

#include "Transformer.h"

namespace
{

RopeConfig ropeConfig(Config const &config)
{
    // checkpoints exported before the RoPE parameters were added to the
    // header have zeros there: use the Llama 3 theta without scaling
    RopeConfig rope{config.ropeTheta, config.ropeFactor, config.ropeLowFreqFactor, config.ropeHighFreqFactor,
                    static_cast<float>(config.ropeOriginalMaxPos)};

    if (rope.theta == 0.0f)
        rope.theta = 500000.0f;

    if (rope.factor < 0.0f ||
        (rope.factor > 0.0f &&
         (rope.lowFreqFactor <= 0.0f || rope.highFreqFactor <= rope.lowFreqFactor || rope.originalMaxPos <= 0.0f)))
        throw std::runtime_error("Invalid RoPE scaling parameters in checkpoint!");

    logger(Logger::INFO) << "RoPE theta: " << rope.theta << ", scaling factor: " << rope.factor
                         << ", low/high freq factor: " << rope.lowFreqFactor << "/" << rope.highFreqFactor
                         << ", original max pos: " << rope.originalMaxPos << std::endl;

    return rope;
}

} // namespace

Transformer::Transformer(Config config)
    : config(std::move(config)),
      tokenEmbeddingTable(config.vocabSize * config.dim),
      layers(config.nLayers, TransformerBlock(config.seqLength, config.dim, config.nHeads, config.nKVHeads,
                                              config.hiddenDim, ropeConfig(config))),
      finalNorm(config.dim),
      output(config.dim, config.vocabSize),
      x(config.dim),
      xb(config.dim),
      logits(config.vocabSize)
{
}

void Transformer::loadWeights(std::ifstream &inputStream)
{
    // the token embedding table is the first tensor in the file. With a
    // shared classifier it is also the weights of the output layer: then
    // only load it there, and use it for the embedding lookup too.
    if (config.sharedClassifier)
        output.loadWeights(inputStream);
    else
        tokenEmbeddingTable.readFromFile(inputStream);

    for (int i = 0; i < config.nLayers; ++i)
        layers[i].loadWeights(inputStream);

    finalNorm.loadWeights(inputStream);

    if (!config.sharedClassifier)
        output.loadWeights(inputStream);
}

Tensor const &Transformer::embeddingTable() const
{
    return config.sharedClassifier ? output.getWeights() : tokenEmbeddingTable;
}

void Transformer::forward(int token, Tensor &logits)
{
    // copy the token embedding into x, dequantizing that row only when needed
    Tensor const &table = embeddingTable();
    size_t const offset = static_cast<size_t>(token) * config.dim;
    float *xf = x.f().data();

    if (table.isQuantizedValid())
    {
        QuantizedTensor const &qt = table.cq();
        for (int i = 0; i < config.dim; i++)
            xf[i] = qt.q[offset + i] * qt.s[(offset + i) / qt.groupSize];
    }
    else
    {
        FloatTensor const &ft = table.cf();
        std::copy(ft.begin() + offset, ft.begin() + offset + config.dim, xf);
    }

    std::reference_wrapper<Tensor> t1 = x;
    std::reference_wrapper<Tensor> t2 = xb;

    // forward all the layers
    for (int l = 0; l < config.nLayers; l++)
    {
        layers[l].forward(t1, t2);
        std::swap(t1, t2);
    }

    // final rmsnorm
    finalNorm.forward(t1, xb);

    // classifier into logits
    output.forward(xb, logits);
}

Config const &Transformer::getConfig() { return config; }
