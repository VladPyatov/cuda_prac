#include <cstdint>
#include "cuda_runtime.h"
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

void canny_cuda(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma);

void build_gaussian_kernel(float *gauss_kernel, int sigma);

__global__ void gaussian_blur_x_kernel(uint8_t *dev_input, uint8_t *dev_blurred_x, int height, int width, int sigma, float *dev_gauss_kernel);

__global__ void gaussian_blur_y_kernel(uint8_t *dev_input, uint8_t *dev_blurred_y, int height, int width, int sigma, float *dev_gauss_kernel);

__global__ void sobel_kernel(const uint8_t *dev_input, float *dev_gradient, uint8_t *dev_direction, int height, int width, int8_t *dev_kernel_x, int8_t *dev_kernel_y);

__global__ void normalization_kernel(const float *dev_input, uint8_t *dev_output, int height, int width, float min, float max);

__global__ void nonmax_suppression_kernel(const uint8_t* dev_input, const uint8_t* dev_direction, uint8_t* dev_suppressed, int height, int width);

__global__ void double_threshold_kernel(uint8_t* dev_input, int height, int width, uint8_t low, uint8_t high);

__global__ void hysteresis_kernel(const uint8_t* input, uint8_t* output, int height, int width, uint8_t low, uint8_t high);

