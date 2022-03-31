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
    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            result[h*width + w] = 0;
        }
    }
    // preprocess image
    uint8_t* blurred_image = (uint8_t*) malloc(height * width * sizeof(uint8_t));
    gaussian_blur(input, blurred_image, height, width, sigma);
    // compute gradients and directions
    uint8_t* gradient = (uint8_t*) calloc(height * width, sizeof(float));
	uint8_t* direction = (uint8_t*) malloc(height * width * sizeof(uint8_t));
    sobel(blurred_image, gradient, direction, height, width);
    // non-maximum suppression
    uint8_t* suppressed = (uint8_t*) calloc(height * width, sizeof(uint8_t));
    nonmax_suppression(gradient, direction, suppressed, height, width);
    //cv::imwrite("4_suppressed.png", cv::Mat(height, width, CV_8UC1, suppressed));
    // hysteresis
    //hysteresis(result, height, width, low_t, high_t);
    hysteresis(suppressed, result, height, width, low_t, high_t);
    // plot blurred
    cv::imwrite("1_blur.png", cv::Mat(height, width, CV_8UC1, blurred_image));
    // plot dir
    cv::imwrite("2_dir.png", cv::Mat(height, width, CV_8UC1, direction));

    // plot grad
    cv::imwrite("3_grad.png", cv::Mat(height, width, CV_8UC1, gradient));

    free(suppressed);
    free(blurred_image);
    free(gradient);
    free(direction);
}


void gaussian_blur(const uint8_t* input, uint8_t* result, int height, int width, int sigma)
{
    //prepare kernel
    int kwidth = 3 * sigma;
    int ksize = kwidth * 2 + 1;
    int pos;
    float kernel[ksize];
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
    //#pragma omp parallel for
    for(int i=0; i<ksize; i++)
    {
        kernel[i] /= sum;
    }

    // x blur
    uint8_t *x_blurred = (uint8_t*) malloc(height * width * sizeof(uint8_t));
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
    for (int h = 0; h<height; h++)
    {
        // setup borders
        //up = max(-h, -kwidth);
        //down = min(kwidth, heigh-h-1);
        up = -h > -kwidth ? -h : -kwidth;
        down = kwidth < height-h-1 ? kwidth : height-h-1;
        for (int w = 0; w<width; w++)
        {
            sum = 0;
            
            for (int i = up; i <= down; i++)
            {
                sum += x_blurred[(h+i)*width + w] * kernel[kwidth + i];
            }
            result[h*width + w] = uint8_t(sum);
        }
    }

    free(x_blurred);
}

// d/dx, d/dy
void sobel(const uint8_t* input, uint8_t* gradient, uint8_t* direction, int height, int width)
{
    const int8_t Gx[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	const int8_t Gy[] = { 1, 2, 1, 0, 0, 0, -1,-2,-1 };

    float grad_x, grad_y, theta;
    float* float_gradient = (float*) calloc(height * width, sizeof(float));
    uint8_t dir, kernel_index, pixel;
    int64_t pixel_position;
    
    #pragma omp parallel for private(grad_x, grad_y, theta, dir, kernel_index, pixel, pixel_position)
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
                for(int x=-1; x<=1; x++, kernel_index++)
                {
                    pixel = input[pixel_position + y*width+x];
                    grad_x += pixel * Gx[kernel_index];
                    grad_y += pixel * Gy[kernel_index];
                }
            }
            float_gradient[pixel_position] = std::sqrt(grad_x*grad_x + grad_y*grad_y);
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

    // perform gradient normalization
    float max_val = float_gradient[0];
    float min_val = max_val;
    #pragma omp parallel for private(pixel) reduction(max:max_val) reduction(min:min_val)
    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            pixel = float_gradient[h*width + w];
            max_val = max_val > pixel ? max_val : pixel;
            min_val = min_val < pixel ? min_val : pixel;
        }
    }
    #pragma omp parallel for
    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            gradient[h*width + w] = (uint8_t) 255*(float_gradient[h*width + w]-min_val)/(max_val-min_val);
        }
    }

    free(float_gradient);
}

void nonmax_suppression(const uint8_t* gradient, const uint8_t* direction, uint8_t* suppressed, int height, int width)
{
    int pixel_position;
    uint8_t left, right, pixel;

    #pragma omp parallel for private(left, right, pixel, pixel_position)
    for(int h=1; h<height-1; h++)
    {
        for(int w=1; w<width-1; w++)
        {
            pixel_position = h*width + w;
            switch (direction[pixel_position])
            {
            case 64:
                left = gradient[pixel_position-1];
                right = gradient[pixel_position+1];
                break;
            case 128:
                left = gradient[pixel_position+width-1];
                right = gradient[pixel_position-width+1];
                break;
            case 192:
                left = gradient[pixel_position+width];
                right = gradient[pixel_position-width];
                break;
            case 255:
                left = gradient[pixel_position-width-1];
                right = gradient[pixel_position+width+1];
                break;
            default:
                break;
            }
            pixel = gradient[pixel_position];
            if (pixel >= left && pixel >= right)
                suppressed[pixel_position] = pixel;
        }
    }
}


// void hysteresis(uint8_t* input, int height, int width, float low_t, float high_t)
// {
//     uint8_t max_val = input[0];
//     uint8_t pixel;

//     #pragma omp parallel for private(pixel) reduction(max:max_val)
//     for (int h = 0; h<height; h++)
//     {
//         for (int w = 0; w<width; w++)
//         {
//             pixel = input[h*width + w];
//             max_val = max_val > pixel ? max_val : pixel;
//         }
//     }
//     printf("%d", max_val);
//     uint8_t high = high_t * max_val;
//     uint8_t low = low_t * high;

//     #pragma omp parallel for private(pixel)
//     for (int h = 1; h<height-1; h++)
//     {
//         for (int w = 1; w<width-1; w++)
//         {
//             pixel = input[h*width + w];
//             input[h*width + w] = pixel > high ? 255 : pixel;
//             input[h*width + w] = pixel < low ? 0 : pixel;
//         }
//     }
//     //cv::imwrite("5_suppressed.png", cv::Mat(height, width, CV_8UC1, input));

//     for (int h = 1; h<height-1; h++)
//     {
//         for (int w = 1; w<width-1; w++)
//         {
//             if(input[h*width + w] >= low && input[h*width + w] <= high)
//                 input[h*width + w] = ((input[h*width + w+1] == 255 || input[h*width + w-1] ||
//                                     input[h*(width+1) + w-1] == 255 || input[h*(width+1) + w] == 255 ||
//                                     input[h*(width+1) + w+1] == 255 || input[h*(width-1) + w-1] == 255 ||
//                                     input[h*(width-1) + w] == 255 || input[h*(width-1) + w+1] == 255)) ? 255 : 0;
//         }
//     }

// }
void hysteresis(uint8_t* input, uint8_t* output, int height, int width, float low_t, float high_t)
{
    uint8_t max_val = input[0];
    uint8_t pixel;

    //#pragma omp parallel for private(pixel) reduction(max:max_val)
    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            pixel = input[h*width + w];
            max_val = max_val > pixel ? max_val : pixel;
        }
    }
    printf("%d", max_val);
    uint8_t high = high_t * max_val;
    uint8_t low = low_t * high;

    //#pragma omp parallel for private(pixel)
    for (int h = 1; h<height-1; h++)
    {
        for (int w = 1; w<width-1; w++)
        {
            pixel = input[h*width + w];
            input[h*width + w] = pixel > high ? 255 : pixel;
            input[h*width + w] = pixel < low ? 0 : pixel;
        }
    }
    //cv::imwrite("5_suppressed.png", cv::Mat(height, width, CV_8UC1, input));
    for (int h = 0; h<height; h++)
    {
        for (int w = 0; w<width; w++)
        {
            output[h*width + w] = input[h*width + w];
        }
    }
    //cv::imwrite("6_suppressed.png", cv::Mat(height, width, CV_8UC1, output));
    for (int h = 1; h<height-1; h++)
    {
        for (int w = 1; w<width-1; w++)
        {
            if(input[h*width + w] >= low && input[h*width + w] <= high)
                output[h*width + w] = ((input[h*width + w+1] == 255 || input[h*width + w-1] ||
                                    input[h*(width+1) + w-1] == 255 || input[h*(width+1) + w] == 255 ||
                                    input[h*(width+1) + w+1] == 255 || input[h*(width-1) + w-1] == 255 ||
                                    input[h*(width-1) + w] == 255 || input[h*(width-1) + w+1] == 255)) ? 255 : 0;
        }
    }
}