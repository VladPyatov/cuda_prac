#include <cstdint>
#include <iostream>
#include <cstdio>
#include <filesystem>
#include <utility>
#include "cuda_runtime.h"
#include "utils.cuh"
#include "layers.cuh"
#include "shared_layers.cuh"

std::pair<float, float> denoise(const uint8_t *input, uint8_t *result, param *weights, int height, int width);

float encoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
float decoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
float refine_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);


