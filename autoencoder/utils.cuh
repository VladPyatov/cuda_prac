#include "cuda_runtime.h"
#include <fstream>
#include <string>
#include <iostream>

void load_weights(float* weight_array, std::string weight_path, unsigned weight_size);

void print_array(float * a, int h, int w, int c);

__global__ void img2float(uint8_t *dev_input, float *dev_output, int height, int width);
__global__ void img2uint(float *dev_input, uint8_t *dev_output, int height, int width);