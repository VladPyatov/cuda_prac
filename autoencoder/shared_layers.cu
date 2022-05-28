#include "shared_layers.cuh"
/*
Shared-memory (weight) implementation of Convolution operation (stride=1) + ReLU
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void SharedConv2DReLU(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int height_cut = kernel_height/2;
    int out_height = in_height - 2*height_cut;
    int width_cut = kernel_width/2;
    int out_width = in_width - 2*width_cut;

    if(z > out_channels-1 || y > out_height-1 || x > out_width-1)
        return;
    
    __shared__ float sdev_weight[S_CHANNELS][S_KERNEL_H][S_KERNEL_W];

    if (threadIdx.x < 3 && threadIdx.y < 3)
    {
            for (int c=0; c<in_channels; c++)
            {
                sdev_weight[c][threadIdx.y][threadIdx.x] = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width + threadIdx.y*kernel_width + threadIdx.x];
            }
    }

    __syncthreads();
    
    float sum = dev_bias[z];
    float value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                value = dev_input[c*in_height*in_width + (y+height_cut + k_h-kernel_height/2)*in_width + x+width_cut + k_w-kernel_width/2];
                sum += value * sdev_weight[c][k_h][k_w];
            }
        }
    }

    dev_output[z*out_height*out_width + y*out_width + x] = sum > 0. ? sum : 0;
}

/*
Shared-memory (weight) implementation of Convolution operation (stride=1) + Sigmoid
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void SharedConv2DSigmoid(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int height_cut = kernel_height/2;
    int out_height = in_height - 2*height_cut;
    int width_cut = kernel_width/2;
    int out_width = in_width - 2*width_cut;

    if(z > out_channels-1 || y > out_height-1 || x > out_width-1)
        return;
    
    __shared__ float sdev_weight[S_CHANNELS][S_KERNEL_H][S_KERNEL_W];

    if (threadIdx.x < 3 && threadIdx.y < 3)
    {
            for (int c=0; c<in_channels; c++)
            {
                sdev_weight[c][threadIdx.y][threadIdx.x] = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width + threadIdx.y*kernel_width + threadIdx.x];
            }
    }

    __syncthreads();
    
    float sum = dev_bias[z];
    float value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                value = dev_input[c*in_height*in_width + (y+height_cut + k_h-kernel_height/2)*in_width + x+width_cut + k_w-kernel_width/2];
                sum += value * sdev_weight[c][k_h][k_w];
            }
        }
    }

    dev_output[z*out_height*out_width + y*out_width + x] = 1/(1+expf(-sum));
}


/*
Shared-memory (feature map) implementation of Convolution operation (stride=1) + ReLU
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void SharedFeatureConv2DReLU(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int height_cut = kernel_height/2;
    int out_height = in_height - 2*height_cut;
    int width_cut = kernel_width/2;
    int out_width = in_width - 2*width_cut;

    if(z > out_channels-1 || y > out_height-1 || x > out_width-1)
        return;
    
    int sdev_h_size = blockDim.y+2*height_cut;
    int sdev_w_size = blockDim.x+2*width_cut;
    __shared__ float sdev_input[S_CHANNELS][S_BLOCK_SIZE_H+2*(int(S_KERNEL_H)/2)][S_BLOCK_SIZE_W+2*(int(S_KERNEL_W)/2)];
    int h_threads = ceil(float(sdev_h_size)/float(blockDim.y));
    int w_threads = ceil(float(sdev_w_size)/float(blockDim.x));
    int y_pos = threadIdx.y * h_threads;
    int x_pos = threadIdx.x * w_threads;
    int s_y = blockDim.y * blockIdx.y + y_pos;
    int s_x = blockDim.x * blockIdx.x + x_pos;
    for (int c=0; c<in_channels; c++)
    {
        for (int h=0; (h < h_threads) && (y_pos+h < sdev_h_size); h++)
        {
            for (int w=0; (w < w_threads) && (x_pos+w < sdev_w_size); w++)
            {
                sdev_input[c][y_pos+h][x_pos+w] = dev_input[c*in_height*in_width + (s_y+h)*in_width + s_x+w];
            }
        }
        
    }
    __syncthreads();
    
    float sum = dev_bias[z];
    float weight, value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                weight = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width+k_h*kernel_width+k_w];
                value = sdev_input[c][threadIdx.y+k_h][threadIdx.x+k_w];
                sum += weight*value;
            }
        }
    }
    dev_output[z*out_height*out_width + y*out_width + x] = sum > 0. ? sum : 0;
}


/*
Shared-memory (feature map) implementation of Convolution operation (stride=1) + Sigmoid
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void SharedFeatureConv2DSigmoid(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int height_cut = kernel_height/2;
    int out_height = in_height - 2*height_cut;
    int width_cut = kernel_width/2;
    int out_width = in_width - 2*width_cut;

    if(z > out_channels-1 || y > out_height-1 || x > out_width-1)
        return;
    
    int sdev_h_size = blockDim.y+2*height_cut;
    int sdev_w_size = blockDim.x+2*width_cut;
    __shared__ float sdev_input[S_CHANNELS][S_BLOCK_SIZE_H+2*(int(S_KERNEL_H)/2)][S_BLOCK_SIZE_W+2*(int(S_KERNEL_W)/2)];
    int h_threads = std::ceil(float(sdev_h_size)/float(blockDim.y));
    int w_threads = std::ceil(float(sdev_w_size)/float(blockDim.x));
    int y_pos = threadIdx.y * h_threads;
    int x_pos = threadIdx.x * w_threads;
    int s_y = blockDim.y * blockIdx.y + y_pos;
    int s_x = blockDim.x * blockIdx.x + x_pos;
    for (int c=0; c<in_channels; c++)
    {
        for (int h=0; (h < h_threads) && (y_pos+h < sdev_h_size); h++)
        {
            for (int w=0; (w < w_threads) && (x_pos+w < sdev_w_size); w++)
            {
                sdev_input[c][y_pos+h][x_pos+w] = dev_input[c*in_height*in_width + (s_y+h)*in_width + s_x+w];
            }
        }
        
    }
    __syncthreads();
    
    float sum = dev_bias[z];
    float weight, value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                weight = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width+k_h*kernel_width+k_w];
                //value = dev_input[c*in_height*in_width + (y+height_cut + k_h-kernel_height/2)*in_width + x+width_cut + k_w-kernel_width/2];
                value = sdev_input[c][threadIdx.y+k_h][threadIdx.x+k_w];
                sum += weight*value;
            }
        }
    }
    dev_output[z*out_height*out_width + y*out_width + x] = 1/(1+expf(-sum));
}