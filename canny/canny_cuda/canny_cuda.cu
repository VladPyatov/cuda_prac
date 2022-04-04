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
    const int8_t sobel_x[] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	const int8_t sobel_y[] = { 1, 2, 1, 0, 0, 0, -1,-2,-1 };

    // device properties
    const dim3 block_size(32, 32);
    const dim3 grid_size(width/32+1, height/32+1);

    // device variables declaration
    float *dev_gauss_kernel;
    uint8_t *dev_input;

    uint8_t *dev_blurred_x, *dev_blurred;

    float *dev_gradient_f;
    uint8_t *dev_direction;
    int8_t *dev_kernel_x, *dev_kernel_y;

    // start timer
	cudaEvent_t global_start, global_stop;
	cudaEventCreate(&global_start);
	cudaEventCreate(&global_stop);
	cudaEventRecord(global_start, 0);

    // kernels computation
    build_gaussian_kernel(gauss_kernel, sigma);
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
    cudaMalloc((void**)&dev_direction, height * width * sizeof(uint8_t));
    // 

    // copy to device from host
    cudaMemcpy(dev_input, input, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gauss_kernel, gauss_kernel, ksize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel_x, sobel_x, 9*sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_kernel_y, sobel_y, 9*sizeof(int8_t), cudaMemcpyHostToDevice);

    gaussian_blur_x_kernel<<<grid_size, block_size>>>(dev_input, dev_blurred_x, height, width, sigma, dev_gauss_kernel);

    gaussian_blur_y_kernel<<<grid_size, block_size>>>(dev_blurred_x, dev_blurred, height, width, sigma, dev_gauss_kernel);;
    sobel<<<grid_size, block_size>>>(dev_blurred, dev_gradient_f, dev_direction, height, width, dev_kernel_x, dev_kernel_y);
    //gradient_normalization

    cudaMemcpy(result, dev_direction, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_gauss_kernel);
    cudaFree(dev_blurred_x);
    cudaFree(dev_blurred);
    cudaFree(dev_gradient_f);
    cudaFree(dev_direction);
    cudaFree(dev_kernel_x);
    cudaFree(dev_kernel_y);


    free(gauss_kernel);

    cudaEventRecord(global_stop, 0);
	cudaEventSynchronize(global_stop);
    float time = 0;
    cudaEventElapsedTime(&time, global_start, global_stop);

    printf("GPU total time: %f ms\n", time);

    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);

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

    if(y >= height || x >= width)
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

    if(y >= height || x >= width)
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

__global__ void sobel(uint8_t *dev_input, float *dev_gradient, uint8_t *dev_direction, int height, int width, int8_t *dev_kernel_x, int8_t *dev_kernel_y)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y >= height || x >= width)
         return;
    
    int kwidth = 1;
    int pixel, dir;
    float grad_x = 0.0;
    float grad_y = 0.0;
    int pixel_position = y*width + x;
    int kernel_index = 0;
    // find kernel limits
    int up = -y > -kwidth ? -y : -kwidth;
    int down = kwidth < height-y-1 ? kwidth : height-y-1;
    int left = -x > -kwidth ? -x : -kwidth;
    int right = kwidth < width-x-1 ? kwidth : width-x-1;
    for(int y_dim=up; y_dim<=down; y_dim++)
    {
        for(int x_dim=left; x_dim<=right; x_dim++, kernel_index++)
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