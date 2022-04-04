#include <cstdint>
#include "cuda_runtime.h"

void canny_cuda(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma);

void build_gaussian_kernel(float *gauss_kernel, int sigma);

__global__ void gaussian_blur_x_kernel(uint8_t *dev_input, uint8_t *dev_blurred_x, int height, int width, int sigma, float *dev_gauss_kernel);

__global__ void gaussian_blur_y_kernel(uint8_t *dev_input, uint8_t *dev_blurred_y, int height, int width, int sigma, float *dev_gauss_kernel);

__global__ void sobel(uint8_t *dev_input, float *dev_gradient, uint8_t *dev_direction, int height, int width, int8_t *dev_kernel_x, int8_t *dev_kernel_y);
