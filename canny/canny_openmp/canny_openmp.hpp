#include <cstdint>
#include "math_constants.h"

void canny_openmp(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma);

void gaussian_blur(const uint8_t* input, uint8_t* result, int height, int width, int sigma);

void sobel(const uint8_t* input, uint8_t* gradient, uint8_t* direction, int height, int width);

void nonmax_suppression(const uint8_t* gradient, const uint8_t* direction, uint8_t* suppressed, int height, int width);

void threshold_limits(uint8_t* input, uint8_t* low, uint8_t* high, int height, int width, float low_t, float high_t);

void double_threshold(uint8_t* input, int height, int width, uint8_t low, uint8_t high);

void hysteresis(const uint8_t* input, uint8_t* output, int height, int width, uint8_t low, uint8_t high);
