#pragma once

#include <fstream>
#include <vector>

#include "Tensor.h"

class RMSNorm
{
public:
    RMSNorm(size_t dim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    size_t dim;
    Tensor weight;
};

class Linear
{
public:
    Linear(size_t inDim, size_t outDim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

    Tensor const &getWeights() const { return weight; }

private:
    size_t inDim;
    size_t outDim;
    Tensor weight;
};

struct RopeConfig
{
    float theta;
    float factor; // 0: no scaling
    float lowFreqFactor;
    float highFreqFactor;
    float originalMaxPos;
};

class CausalAttention
{
public:
    CausalAttention(size_t seqLength, size_t dim, size_t nHeads, size_t nKVHeads, RopeConfig const &rope);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    size_t pos;
    size_t dim;
    size_t nHeads;
    size_t nKVHeads;

    std::vector<float> ropeFreqs; // rotation frequency of each (q0, q1) pair of a head

    Linear wq;
    Linear wk;
    Linear wv;
    Linear wo;

    Tensor query;
    std::vector<Tensor> keyCache;
    std::vector<Tensor> valueCache;

    std::vector<Tensor> att; // buffer for scores/attention values (n_heads, seq_len)

    Tensor xb;
};

class FFN
{
public:
    FFN(size_t dim, size_t hiddenDim);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    size_t dim;
    size_t hiddenDim;

    Linear w1;
    Linear w2;
    Linear w3;

    Tensor hb;
    Tensor hb2;
};

class TransformerBlock
{
public:
    TransformerBlock(size_t seqLength, size_t dim, size_t nHeads, size_t nKVHeads, size_t hiddenDim,
                     RopeConfig const &rope);

    void forward(Tensor &x, Tensor &out);
    void loadWeights(std::ifstream &inputStream);

private:
    RMSNorm attentionNorm;
    CausalAttention attention;
    RMSNorm ffnNorm;
    FFN ffn;

    Tensor xb;
    Tensor xb2;
};
