#include "canny_cuda.hpp"
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void canny_cuda(const uint8_t* input, uint8_t* result, int height, int width, float low_t, float high_t, int sigma)
{
    // kernels
    const int ksize = 3 * sigma * 2 + 1;
    float *gauss_kernel = (float *) malloc(ksize*sizeof(float));
    build_gaussian_kernel(gauss_kernel, sigma);
    const int8_t sobel_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	const int8_t sobel_y[] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

    // device properties
    const dim3 block_size(32, 32);
    const dim3 grid_size(width/32+1, height/32+1);

    // device variables declaration
    float *dev_gauss_kernel;
    uint8_t *dev_input;
    uint8_t *dev_blurred_x, *dev_blurred;
    float *dev_gradient_f;
    uint8_t *dev_direction, *dev_gradient;
    int8_t *dev_kernel_x, *dev_kernel_y;
    uint8_t *dev_suppressed;
    uint8_t high, low;
    uint8_t *dev_result;

    // memory allocation
    // -> for input image
    cudaMalloc((void**)&dev_input, height *width * sizeof(uint8_t));
    // -> for blurring
    cudaMalloc((void**)&dev_gauss_kernel, ksize*sizeof(float));
    cudaMalloc((void**)&dev_blurred_x, height * width * sizeof(uint8_t));
    cudaMalloc((void**)&dev_blurred, height * width * sizeof(uint8_t));
    // -> for gradient and directions
    cudaMalloc((void**)&dev_kernel_x, 9*sizeof(int8_t));
    cudaMalloc((void**)&dev_kernel_y, 9*sizeof(int8_t));
    cudaMalloc((void**)&dev_gradient_f, height * width * sizeof(float));
    cudaMemset(dev_gradient_f, 0.f, height * width * sizeof(float));
    cudaMalloc((void**)&dev_direction, height * width * sizeof(uint8_t));
    cudaMalloc((void**)&dev_gradient, height * width * sizeof(uint8_t));
    cudaMalloc((void**)&dev_result, height * width * sizeof(uint8_t));

    // -> for non_max suppression
    cudaMalloc((void**)&dev_suppressed, height * width * sizeof(uint8_t));
    cudaMemset(dev_suppressed, 0, height * width * sizeof(uint8_t));

    // start timer
	cudaEvent_t global_start, global_stop, copy_start_1, copy_stop_1, copy_start_2, copy_stop_2;
	cudaEventCreate(&global_start);
	cudaEventCreate(&global_stop);
    cudaEventCreate(&copy_start_1);
    cudaEventCreate(&copy_stop_1);
    cudaEventCreate(&copy_start_2);
    cudaEventCreate(&copy_stop_2);
    float time = 0;
    float copy_time_1 = 0;
    float copy_time_2 = 0;
	
    cudaEventRecord(copy_start_1, 0);
    // copy to device from host
    cudaMemcpy(dev_input, input, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gauss_kernel, gauss_kernel, ksize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel_x, sobel_x, 9*sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel_y, sobel_y, 9*sizeof(int8_t), cudaMemcpyHostToDevice);

    //measure copy time
    cudaEventRecord(copy_stop_1, 0);
	cudaEventSynchronize(copy_stop_1);
    cudaEventElapsedTime(&copy_time_1, copy_start_1, copy_stop_1);
    
    cudaEventRecord(global_start, 0);
    // blur
    gaussian_blur_x_kernel<<<grid_size, block_size>>>(dev_input, dev_blurred_x, height, width, sigma, dev_gauss_kernel);
    gaussian_blur_y_kernel<<<grid_size, block_size>>>(dev_blurred_x, dev_blurred, height, width, sigma, dev_gauss_kernel);
    // derivatives
    sobel_kernel<<<grid_size, block_size>>>(dev_blurred, dev_gradient_f, dev_direction, height, width, dev_kernel_x, dev_kernel_y);
    //gradient_normalization
    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(dev_gradient_f);
    thrust::pair<thrust::device_ptr<float>, thrust::device_ptr<float>>  minmax_tuple;
    minmax_tuple = thrust::minmax_element(d_ptr, d_ptr + width*height);
    normalization_kernel<<<grid_size, block_size>>>(dev_gradient_f, dev_gradient, height, width, *(minmax_tuple.first), *(minmax_tuple.second));
    high = high_t * 255;
    low = high * low_t;
    // nonmaximum suppression
    nonmax_suppression_kernel<<<grid_size, block_size>>>(dev_gradient, dev_direction, dev_suppressed, height, width);
    // double threshold
    double_threshold_kernel<<<grid_size, block_size>>>(dev_suppressed, height, width, low, high);
    // hysteresis
    hysteresis_kernel<<<grid_size, block_size>>>(dev_suppressed, dev_result, height, width, low, high);

    cudaEventRecord(global_stop, 0);
	cudaEventSynchronize(global_stop);
    cudaEventElapsedTime(&time, global_start, global_stop);

    cudaEventRecord(copy_start_2, 0);
    cudaMemcpy(result, dev_result, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaEventRecord(copy_stop_2, 0);
	cudaEventSynchronize(copy_stop_2);
    cudaEventElapsedTime(&copy_time_2, copy_start_2, copy_stop_2);

    printf("GPU without copy: %f\n", time);
    printf("GPU with copy: %f\n", time+copy_time_1+copy_time_2);
    printf("GPU copy time: %f\n", copy_time_1+copy_time_2);

    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);
    cudaFree(dev_input);
    cudaFree(dev_gauss_kernel);
    cudaFree(dev_blurred_x);
    cudaFree(dev_blurred);
    cudaFree(dev_gradient_f);
    cudaFree(dev_gradient);
    cudaFree(dev_direction);
    cudaFree(dev_kernel_x);
    cudaFree(dev_kernel_y);
    cudaFree(dev_suppressed);
    cudaFree(dev_result);

    free(gauss_kernel);

}

void build_gaussian_kernel(float *gauss_kernel, int sigma)
{
    // prepare kernel
    int kwidth = 3 * sigma;
    int ksize = kwidth * 2 + 1;
    int pos;
    float sum, value;

    // compute kernel
    for(int i=0; i<ksize; i++)
    {
        pos = i - kwidth;
        value = (float)exp(- float(pos*pos)/float(2*sigma*sigma));
        gauss_kernel[i] = value;
        sum = sum + value;
    }

    // normalize kernel
    for(int i=0; i<ksize; i++)
    {
        gauss_kernel[i] /= sum;
    }
}

__global__ void gaussian_blur_x_kernel(uint8_t *dev_input, uint8_t *dev_blurred_x, int height, int width, int sigma, float *dev_gauss_kernel)
{
    int kwidth = 3 * sigma;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-1 || x > width-1)
         return;

    float sum = 0;
    // setup borders
    //left = max(-x, -kwidth);
    //right = min(kwidth, width-x-1);
    int left = -x > -kwidth ? -x : -kwidth;
    int right = kwidth < width-x-1 ? kwidth : width-x-1;
    for (int i = left; i <= right; i++)
    {
        sum += dev_input[y*width + (x + i)] * dev_gauss_kernel[kwidth + i];
    }
    dev_blurred_x[y*width + x] = uint8_t(sum);
}

__global__ void gaussian_blur_y_kernel(uint8_t *dev_input, uint8_t *dev_blurred_y, int height, int width, int sigma, float *dev_gauss_kernel)
{
    int kwidth = 3 * sigma;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-1 || x > width-1)
         return;

    float sum = 0;

    int up = -y > -kwidth ? -y : -kwidth;
    int down = kwidth < height-y-1 ? kwidth : height-y-1;

            
    for (int i = up; i <= down; i++)
    {
        sum += dev_input[(y+i)*width + x] * dev_gauss_kernel[kwidth + i];
    }
    dev_blurred_y[y*width + x] = uint8_t(sum);
}

__global__ void sobel_kernel(const uint8_t *dev_input, float *dev_gradient, uint8_t *dev_direction, int height, int width, int8_t *dev_kernel_x, int8_t *dev_kernel_y)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-2 || y < 1 || x > width-2 || x < 1)
        return;
    
    //int kwidth = 1;
    int pixel, dir;
    float grad_x = 0.0;
    float grad_y = 0.0;
    int pixel_position = y*width + x;
    int kernel_index = 0;
    // find kernel limits
    //int up = -y > -kwidth ? -y : -kwidth;
    //int down = kwidth < height-y-1 ? kwidth : height-y-1;
    //int left = -x > -kwidth ? -x : -kwidth;
    //int right = kwidth < width-x-1 ? kwidth : width-x-1;
    for(int y_dim=-1; y_dim<=1; y_dim++)
    {
        for(int x_dim=-1; x_dim<=1; x_dim++, kernel_index++)
        {
            pixel = dev_input[pixel_position + y_dim*width+x_dim];
            grad_x += pixel * dev_kernel_x[kernel_index];
            grad_y += pixel * dev_kernel_y[kernel_index];
        }
    }
    dev_gradient[pixel_position] = sqrtf(grad_x*grad_x + grad_y*grad_y);
    // in range [-pi; pi] -> [-180; 180]
    float theta = atan2f(grad_y, grad_x) * 180.0 / M_PI;
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
    
    dev_direction[pixel_position] = dir;
    
}

__global__ void normalization_kernel(const float *dev_input, uint8_t *dev_output, int height, int width, float min, float max)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-1 || x > width-1)
        return;

    dev_output[y*width + x] = (uint8_t) 255*(dev_input[y*width + x]-min)/(max-min);
}

__global__ void nonmax_suppression_kernel(const uint8_t* dev_input, const uint8_t* dev_direction, uint8_t* dev_suppressed, int height, int width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-2 || y < 1 || x > width-2 || x < 1)
        return;

    uint8_t left, right, pixel;
    int pixel_position = y*width + x;

    switch (dev_direction[pixel_position])
    {
    case 64:
        left = dev_input[pixel_position-1];
        right = dev_input[pixel_position+1];
        break;
    case 128:
        left = dev_input[pixel_position+width-1];
        right = dev_input[pixel_position-width+1];
        break;
    case 192:
        left = dev_input[pixel_position+width];
        right = dev_input[pixel_position-width];
        break;
    case 255:
        left = dev_input[pixel_position-width-1];
        right = dev_input[pixel_position+width+1];
        break;
    default:
        break;
    }
    pixel = dev_input[pixel_position];
    if (pixel >= left && pixel >= right)
        dev_suppressed[pixel_position] = pixel;
}

__global__ void double_threshold_kernel(uint8_t* dev_input, int height, int width, uint8_t low, uint8_t high)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-1 || x > width-1)
        return;

    dev_input[y*width + x] = dev_input[y*width + x] > high ? 255 : dev_input[y*width + x];
    dev_input[y*width + x] = dev_input[y*width + x] < low ? 0 : dev_input[y*width + x];

}

__global__ void hysteresis_kernel(const uint8_t* input, uint8_t* output, int height, int width, uint8_t low, uint8_t high)
{
    int h = blockDim.y * blockIdx.y + threadIdx.y;
    int w = blockDim.x * blockIdx.x + threadIdx.x;

    if(h > height-2 || h < 1 || w > width-2 || w < 1)
        return;

    uint8_t pixel = input[h*width + w];
    if(pixel >= low && pixel <= high)
        output[h*width + w] = (input[h*width + w+1] == 255 || input[h*width + w-1] == 255 ||
                            input[h*width + w+width-1] == 255 || input[h*width + w+width] == 255  ||
                            input[h*width + w+width+1] == 255 || input[h*width + w-width-1] == 255  ||
                            input[h*width + w-width] == 255 || input[h*width + w-width+1] == 255 ) ? 255 : 0;
    else
        output[h*width + w] = pixel;
}