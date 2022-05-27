#include <cstdint>
#include <iostream>
#include <cstdio>
#include <filesystem>
#include "cuda_runtime.h"

namespace fs = std::filesystem;

void denoise(const uint8_t *input, uint8_t *result, std::string weights_path, int height, int width);

void encoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
void decoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
void decoder_layer_memory(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);
void refine_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width);


