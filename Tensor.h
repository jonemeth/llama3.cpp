#pragma once

#include <fstream>
#include <mutex>
#include <vector>

#include <iostream>

#include "Logger.h"

template <typename T> struct LoggingAllocator : std::allocator<T>
{
    T *allocate(std::size_t n)
    {
        logger(Logger::DEBUG) << "Allocating " << n << " elements" << std::endl;
        return std::allocator<T>::allocate(n);
    }

    void deallocate(T *p, std::size_t n)
    {
        logger(Logger::DEBUG) << "Deallocating " << n << " elements" << std::endl;
        std::allocator<T>::deallocate(p, n);
    }
};

using FloatTensor = std::vector<float, LoggingAllocator<float>>;
using Int8Tensor = std::vector<int8_t, LoggingAllocator<int8_t>>;

using GroupSize = unsigned int;

struct QuantizedTensor
{
    GroupSize groupSize;
    Int8Tensor q;  // quantized values
    FloatTensor s; // scaling factors
};

class Tensor
{
public:
    Tensor(size_t size);
    Tensor(size_t size, bool initFloat);

    FloatTensor &f();
    FloatTensor const &cf();

    QuantizedTensor &q(GroupSize groupSize);
    QuantizedTensor const &cq() const;
    QuantizedTensor const &cq(GroupSize groupSize);

    size_t size() const;
    bool isQuantizedValid() const;

    void readFromFile(std::ifstream &inputStream);

    void operator=(FloatTensor const &ft);
    void operator=(QuantizedTensor const &qt);

private:
    void readFloatFromFile(std::ifstream &inputStream);
    void readQuantizedFromFile(std::ifstream &inputStream, GroupSize groupSize);

    void ensureFloat();
    void ensureQuantized(GroupSize groupSize);

private:
    size_t size_;

    bool isFloatValid_;
    bool isQuantizedValid_;

    FloatTensor floatTensor;
    QuantizedTensor quantizedTensor;
};