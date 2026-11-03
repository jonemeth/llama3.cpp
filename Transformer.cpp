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
    // Read tokenEmbeddingTable which might serve as the weights of
    // the classifier when sharedClassifier is true.
    Tensor tet(config.vocabSize * config.dim);
    tet.readFromFile(inputStream);

    // We always use f32 for tokenEmbeddingTable, but we might use the
    // quantized values for the classifier so keep them valid by
    // using the const version of f().
    this->tokenEmbeddingTable = tet.cf();

    for (int i = 0; i < config.nLayers; ++i)
        layers[i].loadWeights(inputStream);

    finalNorm.loadWeights(inputStream);

    if (config.sharedClassifier)
        if (tet.isQuantizedValid())
            output.setWeights(tet.cq());
        else
            output.setWeights(tet.f());
    else
        output.loadWeights(inputStream);
}

void Transformer::forward(int token, Tensor &logits)
{
    // copy the token embedding into x
    std::copy(tokenEmbeddingTable.data() + token * config.dim, tokenEmbeddingTable.data() + (1 + token) * config.dim,
              x.f().data());

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
