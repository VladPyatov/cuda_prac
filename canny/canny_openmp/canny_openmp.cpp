#define _USE_MATH_DEFINES
#include "canny_openmp.hpp"
#include "omp.h"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <opencv2/imgcodecs.hpp>


void canny_openmp(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma)
{   
    // preprocess image
    //cv::Mat blurred_image(height, width, CV_8UC1);
    uint8_t* blurred_image = (uint8_t*) malloc(height * width);
    gaussian_blur(input, blurred_image, height, width, sigma);
    // compute gradients and directions
    float* gradient = (float*) calloc(height * width, sizeof(float));
	uint8_t* direction = (uint8_t*) malloc(height * width);
    sobel(blurred_image, gradient, direction, height, width);

    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            result[h*width+w] = gradient[h*width+w];
            
        }
    }
    //     std::cout << blurred_image[h*width+width-1] << std::endl;
    // }
    delete [] blurred_image;

}


void gaussian_blur(const uint8_t* input, uint8_t* result, int height, int width, int sigma)
{
    //prepare kernel
    int kwidth = 3 * sigma;
    int ksize = kwidth * 2 + 1;
    int pos;
    float *kernel = new float[ksize];
    float sum, value;

    #pragma omp parallel for private(pos,value) reduction(+:sum)
    for(int i=0; i<ksize; i++)
    {
        pos = i - kwidth;
        value = (float)exp(- float(pos*pos)/float(2*sigma*sigma));
        kernel[i] = value;
        sum = sum + value;
    }

    // normalize kernel
    #pragma omp parallel for
    for(int i=0; i<ksize; i++)
    {
        kernel[i] /= sum;
    }

    // x blur
    uint8_t *x_blurred = new uint8_t[height*width];
    int left, right;

    #pragma omp parallel for private(left, right, sum)
    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            sum = 0;
            // setup borders
            //left = max(-w, -kwidth);
            //right = min(kwidth, width-w-1);
            left = -w > -kwidth ? -w : -kwidth;
            right = kwidth < width-w-1 ? kwidth : width-w-1;
            for (int i = left; i <= right; i++)
            {
                sum += input[h*width + (w + i)] * kernel[kwidth + i];
            }
            x_blurred[h*width + w] = uint8_t(sum);
        }
    }

    //y blur
    int up, down;
    #pragma omp parallel for private(up,down,sum)
    for (int w = 0; w<width; w++)
    {
        for (int h = 0; h<height; h++)
        {
            sum = 0;
            // setup borders
            //up = max(-h, -kwidth);
            //down = min(kwidth, heigh-h-1);
            up = -h > -kwidth ? -h : -kwidth;
            down = kwidth < height-h-1 ? kwidth : height-h-1;
            for (int i = up; i <= down; i++)
            {
                sum += x_blurred[(h+i)*width + w] * kernel[kwidth + i];
            }
            result[h*width + w] = uint8_t(sum);
        }
    }

    delete [] x_blurred;
    delete [] kernel;
}


void sobel(const uint8_t* input, float* gradient, uint8_t* direction, int height, int width)
{
    const int8_t Gx[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	const int8_t Gy[] = { 1, 2, 1, 0, 0, 0, -1,-2,-1 };

    double grad_x, grad_y, magnitude, theta;
    uint8_t dir, kernel_index, pixel;
    int64_t pixel_position;


    for(int h=1; h<height-1; h++)
    {
        for(int w=1; w<width-1; w++)
        {
            grad_x = 0.0;
            grad_y = 0.0;
            pixel_position = h*width + w;
            kernel_index = 0;
            for(int y=-1; y<=1; y++)
            {
                for(int x=-1; x<=1; x++)
                {
                    pixel = input[pixel_position + y*width+x];
                    grad_x += pixel * Gx[kernel_index];
                    grad_y += pixel * Gy[kernel_index];
                }
            }
            gradient[pixel_position] = std::sqrt(grad_x*grad_x + grad_y*grad_y);
            // in range [-pi; pi] -> [-180; 180]
            theta = std::atan2(grad_y, grad_x) * 180.0 / M_PI;

            // 0˚
            if ((theta <= 22.5 && theta >= -22.5) || (theta <= -157.5) && (theta >= 157.5))
				dir = 64;
            // 45˚
			else if ((theta > 22.5 && theta <= 67.5) || (theta > -157.5 && theta <= -112.5))
                dir = 128;
            // 90˚
			else if ((theta > 67.5 && theta <= 112.5) || (theta >= -112.5 && theta < -67.5))
			    dir = 192;
            // 135˚
			else
				dir = 255;
            
            direction[pixel_position] = dir;
        }
    }
}

