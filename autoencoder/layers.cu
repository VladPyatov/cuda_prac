/*
Padding with zeros operation
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_output - output tensor of shape C_in x (H+up+down) x (W+left+right)
    @param in_channels, height, width - input tensor shape
    @param up, down, left, right - padding size
*/
__global__ void ZeroPad2D(float *dev_input, float *dev_output, int in_channels, int height, int width, int up, int down, int left, int right)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (z > in_channels - 1 || y > height + up + down - 1 || x > width + left + right -1)
        return;
    // pixel position
    else if((y > up - 1)  &&  (y < height + up) && (x > left - 1)  &&  (x < width + left))
    {
        dev_output[z*(height+up+down)*(width+left+right) + y*(width+left+right) + x] = dev_input[z*height*width + (y-up)*width + x - left];
    }
    // zero position
    else
    {
        dev_output[z*(height+up+down)*(width+left+right)+ y*(width+left+right) + x] = 0.;
    }
}


/*
Convolution operation (stride=1)
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void Conv2D(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
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

    float sum = 0;
    float weight, value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                weight = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width+k_h*kernel_width+k_w];
                value = dev_input[c*in_height*in_width + (y+height_cut + k_h-kernel_height/2)*in_width + x+width_cut + k_w-kernel_width/2];
                sum += weight*value;
            }
        }
    }
    dev_output[z*out_height*out_width + y*out_width + x] = sum + dev_bias[z];
}


/*
Convolution operation (stride=1) + ReLU
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void Conv2DReLU(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
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

    float sum = dev_bias[z];
    float weight, value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                weight = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width+k_h*kernel_width+k_w];
                value = dev_input[c*in_height*in_width + (y+height_cut + k_h-kernel_height/2)*in_width + x+width_cut + k_w-kernel_width/2];
                sum += weight*value;
            }
        }
    }
    dev_output[z*out_height*out_width + y*out_width + x] = sum > 0. ? sum : 0;
}


/*
Convolution operation (stride=1) + Sigmoid
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_weight - conv weight of shape C_out x C_in x H x W
    @param dev_bias - conv bias of shape C_out
    @param dev_output - output tensor of shape C_out x H x W
    @param in_channels, in_height, in_width - input tensor shape
    @param out_channels - # of output channels
    @param kernel_height, kernel_width - kernel size
*/
__global__ void Conv2DSigmoid(float *dev_input, float *dev_weight, float *dev_bias, float *dev_output, int in_channels, int out_channels, int in_height, int in_width, int kernel_height, int kernel_width)
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

    float sum = dev_bias[z];
    float weight, value;

    for(int c=0; c<in_channels; c++)
    {
        for(int k_h=0; k_h<kernel_height; k_h++)
        {
            for(int k_w=0; k_w<kernel_width; k_w++)
            {
                weight = dev_weight[z*kernel_height*kernel_width*in_channels + c*kernel_height*kernel_width+k_h*kernel_width+k_w];
                value = dev_input[c*in_height*in_width + (y+height_cut + k_h-kernel_height/2)*in_width + x+width_cut + k_w-kernel_width/2];
                sum += weight*value;
            }
        }
    }
    dev_output[z*out_height*out_width + y*out_width + x] = 1/(1+expf(-sum));
}


/*
Max Pooling operation (general case - any stride and any kernel)
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_output - output tensor of shape C_in x H/stride_h x W/stride_w
    @param in_channels, in_height, in_width - input tensor shape
    @param kernel_height, kernel_width - kernel size
    @param stride_h, stride_w - kernel stride
*/
__global__ void MaxPool2D(float *dev_input, float *dev_output, int in_channels, int in_height, int in_width, int kernel_height, int kernel_width, int stride_h, int stride_w)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int out_height = in_height/stride_h;
    int out_width = in_width/stride_w;

    if(z > in_channels-1 || y > out_height-1 || x > out_width-1)
        return;
    
    int pixel_position = z*in_height*in_width + y*stride_h*in_width + x*stride_w;
    float max = dev_input[pixel_position];
    float pixel;

    for(int h = 0; h<kernel_height && (y*stride_h+h < in_height); h++)
    {
        for(int w = 0; w<kernel_width && (x*stride_w+w < in_width); w++)
        {
            pixel_position = z*in_height*in_width + (y*stride_h + h)*in_width + x*stride_w+w;
            pixel = dev_input[pixel_position];
            //printf("pixel=%f\n", pixel);
            max = pixel > max ? pixel : max;
        }
    }

    dev_output[z*out_height*out_width + y*out_width + x] = max;
}


/*
Chess Upsampling operation - makes chessboard from feature channels
    @param dev_input - input tensor of shape C_in x H x W
    @param dev_output - output tensor of shape C_in x (H + H-1) x (W + W-1)
    @param in_channels, in_height, in_width - input tensor shape
*/
__global__ void ChessUpsample2D(float *dev_input, float *dev_output, int in_channels, int in_height, int in_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    int out_height = in_height + in_height-1;
    int out_width = in_width + in_width-1;

    if(z > in_channels-1 || y > out_height-1 || x > out_width-1)
        return;

    if (x%2==0 && y%2==0)
    {
        dev_output[z*out_height*out_width + y*out_width + x] = dev_input[z*in_height*in_width + (y/2)*in_width + x/2];
    }
    else
    {
        dev_output[z*out_height*out_width + y*out_width + x] = 0.;
    }
}


/*
Weight tensor flipping operation - flips every H x W weight map up-down and left-right
    @param dev_input_weight - input tensor of shape C_in x H x W
    @param out_channels - # of output channels
    @param in_channels, in_height, in_width - input tensor shape
*/
__global__ void FlipWeight2D(float *dev_input_weight, int out_channels, int in_channels, int in_height, int in_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if(z > out_channels-1 || y > in_height/2 || (y == in_height/2) && (x > in_width/2)  || x > in_width-1)
        return;
    
    float temp;
    for(int c=0; c<in_channels; c++)
    {
        temp = dev_input_weight[z*in_channels*in_height*in_width + c*in_height*in_width + y*in_width + x];
        dev_input_weight[z*in_channels*in_height*in_width + c*in_height*in_width + y*in_width + x] = dev_input_weight[z*in_channels*in_height*in_width + c*in_height*in_width + (in_height-y-1)*in_width + in_width-x-1];
        dev_input_weight[z*in_channels*in_height*in_width + c*in_height*in_width + (in_height-y-1)*in_width + in_width-x-1] = temp;
    }
}

/*
Kernel transposition operation - works like FlipWeight2D + transposes 0 and 1 dimensions
    @param dev_input - input tensor of shape C_in x C_out x H x W
    @param dev_output - output tensor of shape C_out x C_in x H x W
    @param out_channels - # of output channels
    @param in_channels, in_height, in_width - input tensor shape
*/
__global__ void TransposeKernel(float *dev_input, float *dev_output, int out_channels, int in_channels, int in_height, int in_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (z > out_channels*in_channels - 1 || y > in_height - 1 || x > in_width -1)
        return;
    
    int feature_type = z % out_channels;
    int feature_position_start = in_channels * feature_type;
    int position_shift = z/out_channels;

    dev_output[(feature_position_start+position_shift)*in_height*in_width + (in_height-y-1)*in_width + in_width-x-1] = dev_input[z*in_height*in_width + y*in_width + x];
    
}

/*
ReLU activation function
    @param dev_input - input tensor of shape C_in x H x W
    @param in_channels, in_height, in_width - input tensor shape
*/
__global__ void ReLU(float *dev_input, int in_channels, int in_height, int in_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (z > in_channels - 1 || y > in_height - 1 || x > in_width - 1)
        return;
    
    float pixel = dev_input[z*in_height*in_width + y*in_width + x];
    dev_input[z*in_height*in_width + y*in_width + x] =  pixel > 0. ? pixel : 0.;
}

/*
Sigmoid activation function
    @param dev_input - input tensor of shape C_in x H x W
    @param in_channels, in_height, in_width - input tensor shape
*/
__global__ void Sigmoid(float *dev_input, int in_channels, int in_height, int in_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (z > in_channels - 1 || y > in_height - 1 || x > in_width -1)
        return;
    
    float pixel = dev_input[z*in_height*in_width + y*in_width + x];
    dev_input[z*in_height*in_width + y*in_width + x] =  1/(1+expf(-pixel));
}