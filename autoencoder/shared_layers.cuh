#define S_CHANNELS 32
#define S_KERNEL_W 3
#define S_KERNEL_H 3
#define S_BLOCK_SIZE_H 16
#define S_BLOCK_SIZE_W 16

#include "cuda_runtime.h"

__global__ void SharedConv2DReLU(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width);
__global__ void SharedConv2DSigmoid(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width);

__global__ void SharedFeatureConv2DReLU(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width);
__global__ void SharedFeatureConv2DSigmoid(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width);