#include <cstdint>
#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <filesystem>
#include "cuda_runtime.h"
namespace fs = std::filesystem;

void denoise(const uint8_t *input, uint8_t *result, std::string weights_path, int height, int width);

void load_weights(float* weight_array, std::string weight_path, unsigned weight_size);

void print_array(float * a, int h, int w, int c);

__global__ void img2float(uint8_t *dev_input, float *dev_output, int height, int width);

__global__ void img2uint(float *dev_input, uint8_t *dev_output, int height, int width);

__global__ void ZeroPad2D(float *dev_input, float *dev_output, int in_channels, int height, int width, int up=0, int down=0, int left=0, int right=0);
__global__ void Conv2D(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width);
__global__ void MaxPool2D(float *dev_input, float *dev_output, int in_channels, int in_height, int in_width, int kernel_height, int kernel_width, int stride_h, int stride_w);
__global__ void ChessUpsample2D(float *dev_input, float *dev_output, int in_channels, int in_height, int in_width);
__global__ void FlipWeight2D(float *dev_input_weight, int out_channels, int in_channels, int in_height, int in_width);
__global__ void ReLU(float *dev_input, int in_channels, int in_height, int in_width);
__global__ void Sigmoid(float *dev_input, int in_channels, int in_height, int in_width);
__global__ void TransposeKernel(float *dev_input, float *dev_output, int out_channels, int in_channels, int in_height, int in_width);

void encoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
void decoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
void decoder_layer_memory(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
void refine_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);


