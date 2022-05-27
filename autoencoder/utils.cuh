#include "cuda_runtime.h"
#include <fstream>
#include <string>
#include <iostream>

struct param
{
    float *weight;
    float *bias;
    int dim_0, dim_1, dim_2, dim_3;

    param(float *w, float *b, int d_0, int d_1, int d_2, int d_3)
    {
        weight = w;
        bias = b;
        dim_0 = d_0;
        dim_1 = d_1;
        dim_2 = d_2;
        dim_3 = d_3;
    }
};

void load_weights(float* weight_array, std::string weight_path, unsigned weight_size);

void print_array(float * a, int h, int w, int c);

__global__ void img2float(uint8_t *dev_input, float *dev_output, int height, int width);
__global__ void img2uint(float *dev_input, uint8_t *dev_output, int height, int width);