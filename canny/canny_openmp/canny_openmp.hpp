#include <cstdint>

void canny_openmp(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma);

void gaussian_blur(const uint8_t* input, uint8_t* result, int height, int width, int sigma);

void sobel(const uint8_t* input, float* gradient, uint8_t* direction, int height, int width);