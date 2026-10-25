#pragma once

#include <memory>
#include <vector>

#include "Tensor.h"

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

class Sampler
{
public:
    using SP = std::shared_ptr<Sampler>;

public:
    virtual size_t sample(FloatTensor const &logits) = 0;
};

class ArgmaxSampler : public Sampler
{
public:
    size_t sample(FloatTensor const &logits);
};

class SimpleSampler : public Sampler
{
public:
    SimpleSampler(float temperature, unsigned long long rngSeed);
    size_t sample(FloatTensor const &logits);

private:
    float temperature;
    unsigned long long rngState;

    FloatTensor probs;
};

class NucleusSampler : public Sampler
{
public:
    NucleusSampler(size_t dim, float temperature, float topP, unsigned long long rngSeed);
    size_t sample(FloatTensor const &logits);

private:
    typedef struct
    {
        float prob;
        size_t index;
    } ProbIndex; // struct used when sorting probabilities during top-p sampling

    size_t sampleFromDistribution(ProbIndex *first, ProbIndex *last, float cumulativeProb);

private:
    float temperature;
    float topP;
    unsigned long long rngState;

    FloatTensor probs;
    std::vector<ProbIndex> probIndex; // buffer used in top-p sampling
};
