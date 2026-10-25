#include <algorithm>
#include <cmath>
#include <numeric>

#include "Sampler.h"

void softmax(FloatTensor const &logits, FloatTensor &dst, float temperature)
{
    dst.resize(logits.size());
    size_t dim = dst.size();

    // temperature scaling and
    // find max value (for numerical stability)
    float maxVal = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < dim; ++i)
    {
        dst[i] = logits[i] / temperature;
        if (dst[i] > maxVal)
            maxVal = dst[i];
    }

    // exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i)
    {
        dst[i] = std::exp(dst[i] - maxVal);
        sum += dst[i];
    }

    // normalize
    for (size_t i = 0; i < dim; ++i)
        dst[i] /= sum;
}

unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state)
{ // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

size_t sampleFromDistribution(FloatTensor::const_iterator first, FloatTensor::const_iterator last,
                              unsigned long long *rng_state)
{
    float coin = random_f32(rng_state);
    size_t const n = std::distance(first, last);

    float cdf = 0.0f;
    for (size_t i = 0; first != last; i++, first++)
    {
        cdf += *first;
        if (coin < cdf)
            return i;
    }
    return n - 1; // in case of rounding errors
}

size_t sampleFromDistribution(FloatTensor const &probs, unsigned long long *rng_state)
{
    return sampleFromDistribution(probs.begin(), probs.end(), rng_state);
}

size_t ArgmaxSampler::sample(FloatTensor const &logits)
{
    // greedy argmax sampling: take the token with the highest probability
    return std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
}

SimpleSampler::SimpleSampler(float temperature, unsigned long long rngSeed)
    : temperature(temperature),
      rngState(rngSeed)
{
}

size_t SimpleSampler::sample(FloatTensor const &logits)
{
    softmax(logits, probs, temperature);
    return sampleFromDistribution(probs, &rngState);
}

NucleusSampler::NucleusSampler(size_t dim, float temperature, float topP, unsigned long long rngSeed)
    : temperature(temperature),
      topP(topP),
      rngState(rngSeed),
      probs(dim),
      probIndex(dim)
{
}

size_t NucleusSampler::sample(FloatTensor const &logits)
{
    softmax(logits, probs, temperature);

    auto probindexFirst = probIndex.data();
    auto probindexLast = probIndex.data();

    const float cutoff = (1.0f - topP) / (probs.size() - 1);
    for (size_t i = 0; i < probs.size(); i++)
        if (probs[i] >= cutoff)
            *probindexLast++ = ProbIndex{probs[i], i};

    std::sort(probindexFirst, probindexLast,
              [](auto const &first, auto const &second) { return first.prob > second.prob; });

    float cumulativeProb = 0.0f;
    auto it = probindexFirst;
    for (; it != probindexLast; ++it)
        if ((cumulativeProb += it->prob) > topP)
            break; // we've exceeded topp by including last_idx
    if (it == probindexLast)
        --it; // rounding: never exceeded topP, so use all candidates

    size_t ix = sampleFromDistribution(probindexFirst, it + 1, cumulativeProb);

    return probIndex[ix].index;
}

size_t NucleusSampler::sampleFromDistribution(ProbIndex *first, ProbIndex *last, float cumulativeProb)
{
    float coin = random_f32(&rngState) * cumulativeProb;

    float cdf = 0.0f;
    auto it = first;
    for (size_t i = 0; it != last; i++, it++)
    {
        cdf += it->prob;
        if (coin < cdf)
            return i;
    }
    return std::distance(first, last) - 1; // in case of rounding errors
}