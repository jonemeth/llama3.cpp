#include <cmath>
#include <stdexcept>

#include "Tensor.h"

namespace detail
{

inline void dequantize(FloatTensor &dest, QuantizedTensor const &source)
{
    for (size_t i = 0; i < dest.size(); i++)
        dest[i] = source.q[i] * source.s[i / source.groupSize];
}

inline void quantize(QuantizedTensor &qx, FloatTensor const &x, GroupSize groupSize)
{
    int num_groups = x.size() / groupSize;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++)
    {

        // find the max absolute value in the current group
        float wmax = 0.0f;
        for (size_t i = 0; i < groupSize; i++)
        {
            float val = std::abs(x[group * groupSize + i]);
            if (val > wmax)
                wmax = val;
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx.s[group] = scale;
        float const invScale = scale != 0.0f ? 1.0f / scale : 0.0f;

        // calculate and write the quantized values
        for (size_t i = 0; i < groupSize; i++)
        {
            float quant_value = x[group * groupSize + i] * invScale;         // scale
            int8_t quantized = static_cast<int8_t>(std::round(quant_value)); // round and clamp
            qx.q[group * groupSize + i] = quantized;
        }
    }
}

} // namespace detail

Tensor::Tensor(size_t size)
    : size_(size),
      isFloatValid_(false),
      isQuantizedValid_(false)
{
}

Tensor::Tensor(size_t size, bool initFloat)
    : Tensor(size)
{
    if (initFloat)
        ensureFloat();
}

FloatTensor &Tensor::f()
{
    ensureFloat();
    isQuantizedValid_ = false;
    return floatTensor;
}

FloatTensor const &Tensor::cf()
{
    ensureFloat();
    return floatTensor;
}

void Tensor::ensureFloat()
{
    if (!isFloatValid_)
    {
        floatTensor.resize(size_);

        if (isQuantizedValid_)
            detail::dequantize(floatTensor, quantizedTensor);
        isFloatValid_ = true;
    }
}

QuantizedTensor &Tensor::q(GroupSize groupSize)
{
    if (isQuantizedValid_ && groupSize != quantizedTensor.groupSize)
        throw std::runtime_error("Trying to re-quantize Tensor. It would result extremely slow performance!");

    ensureQuantized(groupSize);

    isFloatValid_ = false;
    return quantizedTensor;
}

QuantizedTensor const &Tensor::cq() const
{
    if (!isQuantizedValid_)
        throw std::runtime_error("Trying to access invalid quantized tensor");
    return quantizedTensor;
}

QuantizedTensor const &Tensor::cq(GroupSize groupSize)
{
    ensureQuantized(groupSize);
    return quantizedTensor;
}

void Tensor::ensureQuantized(GroupSize groupSize)
{
    if (!isQuantizedValid_)
    {
        quantizedTensor.q.resize(size_);
        quantizedTensor.s.resize(size_ / groupSize);
        quantizedTensor.groupSize = groupSize;

        if (isFloatValid_)
            detail::quantize(quantizedTensor, floatTensor, groupSize);
        isQuantizedValid_ = true;
    }
}

size_t Tensor::size() const { return size_; }

bool Tensor::isQuantizedValid() const { return isQuantizedValid_; }

void Tensor::readFromFile(std::ifstream &inputStream)
{
    GroupSize groupSize = 0;
    inputStream.read((char *)&groupSize, sizeof(groupSize));

    if (0 == groupSize)
        readFloatFromFile(inputStream);
    else
        readQuantizedFromFile(inputStream, groupSize);
}

void Tensor::readFloatFromFile(std::ifstream &inputStream)
{
    floatTensor.resize(size_);
    inputStream.read((char *)floatTensor.data(), size_ * sizeof(float));

    isFloatValid_ = true;
    isQuantizedValid_ = false;
}

void Tensor::readQuantizedFromFile(std::ifstream &inputStream, GroupSize groupSize)
{
    if (size_ % groupSize != 0)
    {
        throw std::runtime_error("Tensor size should be divisible by group-size!");
    }
    quantizedTensor.groupSize = groupSize;
    quantizedTensor.q.resize(size_);
    quantizedTensor.s.resize(size_ / groupSize);

    inputStream.read((char *)quantizedTensor.q.data(), size_ * sizeof(*quantizedTensor.q.data()));
    inputStream.read((char *)quantizedTensor.s.data(), size_ / groupSize * sizeof(*quantizedTensor.s.data()));

    isFloatValid_ = false;
    isQuantizedValid_ = true;
}

void Tensor::operator=(FloatTensor const &ft)
{
    size_ = ft.size();
    floatTensor = ft;
    isFloatValid_ = true;
    isQuantizedValid_ = false;
}

void Tensor::operator=(QuantizedTensor const &qt)
{
    size_ = qt.q.size();
    quantizedTensor = qt;
    isFloatValid_ = false;
    isQuantizedValid_ = true;
}
