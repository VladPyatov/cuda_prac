#include "canny_cuda.hpp"

void canny_cuda(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma);