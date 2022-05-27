#include "cuda_runtime.h"

__global__ void ZeroPad2D(float *dev_input, float *dev_output, int in_channels, int height, int width, int up=0, int down=0, int left=0, int right=0);
__global__ void Conv2D(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width);
__global__ void MaxPool2D(float *dev_input, float *dev_output, int in_channels, int in_height, int in_width, int kernel_height, int kernel_width, int stride_h, int stride_w);
__global__ void ChessUpsample2D(float *dev_input, float *dev_output, int in_channels, int in_height, int in_width);
__global__ void FlipWeight2D(float *dev_input_weight, int out_channels, int in_channels, int in_height, int in_width);
__global__ void ReLU(float *dev_input, int in_channels, int in_height, int in_width);
__global__ void Sigmoid(float *dev_input, int in_channels, int in_height, int in_width);
__global__ void TransposeKernel(float *dev_input, float *dev_output, int out_channels, int in_channels, int in_height, int in_width);