#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>

#include "Tensor.h"
#include "layers.h"

struct Config
{
    int dim;       // transformer dimension
    int hiddenDim; // for ffn layers
    int nLayers;   // number of layers
    int nHeads;    // number of query heads
    int nKVHeads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocabSize; // vocabulary size, usually 4096 (byte-level)
    int seqLength; // max sequence length
    uint8_t sharedClassifier;
    uint8_t padding[3];

    // RoPE parameters. Older checkpoints have zeros here, see ropeConfig().
    float ropeTheta;          // base of the rotation frequencies (0: 500000)
    float ropeFactor;         // Llama 3.1 style frequency scaling (0: no scaling)
    float ropeLowFreqFactor;  // wavelengths above originalMaxPos/lowFreqFactor are fully scaled
    float ropeHighFreqFactor; // wavelengths below originalMaxPos/highFreqFactor are not scaled
    int ropeOriginalMaxPos;   // context length the model was originally trained with
};

// The Config is stored right after the magic number and the version in the
// 256 bytes long header, so the RoPE parameters start at file offset 40.
static_assert(offsetof(Config, ropeTheta) == 32);
static_assert(8 + sizeof(Config) <= 256);

class Transformer
{
public:
    Transformer(Config config);

    void loadWeights(std::ifstream &inputStream);
    void forward(int token, Tensor &logits);

    Config const &getConfig();

private:
    Config config;

    FloatTensor tokenEmbeddingTable; // (vocab_size, dim)

    std::vector<TransformerBlock> layers;
    RMSNorm finalNorm;
    Linear output;

    Tensor x;
    Tensor xb;
    Tensor logits;
};