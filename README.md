# llama3.cpp
Llama3 inference in pure C++.

This is a C++ port of [llama3.c](https://github.com/jameswdelancey/llama3.c) by [James Delancey](https://github.com/jameswdelancey), which is a modified version of [llama2.c](https://github.com/karpathy/llama2.c): by [Andrej Karpathy](https://github.com/karpathy).

Unlike the single-file C implementation, here the source code is split into multiple source files. Algorithms from the standard C++ library have been applied in many cases, but most of the algorithms and methods can be identified with their counterpart in the original C version. The original comments have also been kept unchanged where it was appropriate.

However, in the case of llama3.cpp, the same executable can be used with both float32 and int8 weights: the `Tensor` class automatically converts between float and quantized values, while the `Linear` layer executes quantized matrix multiplication when its weight tensor contains quantized values. Note that it required to change the order in which `export.py` exports the model weights.

llama3.cpp uses the [argparse](https://github.com/morrisfranken/argparse) library by [Morris Franken](https://github.com/morrisfranken).


## Usage
The code is written in C++20 and tested under Linux.

### Build
After cloning the repository, the executable can be compiled either using `cmake` or by:
```
g++ -std=c++20 -Wall -Wextra -Werror -Ofast -fopenmp -march=native *.cpp -o llama3
```

### Convert models

The first step is to download one of the Llama3 models by following the instructions on the [Llama repository](https://github.com/meta-llama/llama-models), or by cloning the [huggingface]()https://huggingface.co/meta-llama Llama repositories, e.g.:
```
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
```

To use the Llama3 checkpoints, they need to be converted into a format that can be used with llama3.cpp. To export float32 weights using:

```
mkdir models
python3 export.py ./models/Llama3.2-1B.bin --meta-llama <PATH-TO>/Llama3.2-1B/ --rope-preset llama3.2
```

or quantized weights with the `--quantize` flag:
```
python3 export.py ./models/Llama3.2-1B-q80.bin --meta-llama <PATH-TO>/Llama3.2-1B/ --rope-preset llama3.2 --quantize
```

To convert models downloaded from huggingface use the `--hf` argument instead of `--meta-llama`.

The RoPE parameters (theta and the Llama 3.1 style frequency scaling) are stored in the header of the exported file. They are read from the `config.json` of huggingface models, but Meta's `params.json` doesn't contain the scaling factor: `--meta-llama` exports assume Llama 3.1 (factor 8), so use `--rope-preset llama3.2` (factor 32) for Llama 3.2 models.


The following models have been tried:
 * Llama3.1-8B and Llama3.1-8B-Instruct
 * Llama3.2-1B and Llama3.2-1B-Instruct
 * Llama3.2-3B and Llama3.2-3B-Instruct

### Run llama3
The compiled executable can be used with both float32 and quantized models, for example:
```
./llama3 ./models/Llama3.1-8B-q80.bin -i "<HERE GOES THE PROMPT>"
```

For chat mode set the `-m chat` argument, and use an "Instruct" model:
```
./llama3 ./models/Llama3.1-8B-Instruct-q80.bin -m chat
```
