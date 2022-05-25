#include "cnn_autoencoder.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void denoise(const uint8_t* input, uint8_t* result, std::string weights_path, int height, int width)
{
    fs::path weights_path_object = fs::path(weights_path);

    // allocate memory for encoder weights
    float enc_conv1_weight[32*1*3*3];
    float enc_conv1_bias[32];
    float enc_conv2_weight[32*32*3*3];
    float enc_conv2_bias[32];
    // allocate memory for decoder weights
    float dec_tconv1_weight[32*32*3*3];
    float dec_tconv1_bias[32];
    float dec_tconv2_weight[32*32*3*3];
    float dec_tconv2_bias[32];
    float dec_conv1_weight[1*32*3*3];
    float dec_conv1_bias[1];

    // load encoder weights
    load_weights(enc_conv1_weight, weights_path_object/"enc_conv1_weight.bin", 32*1*3*3);
    load_weights(enc_conv1_bias, weights_path_object/"enc_conv1_bias.bin", 32);
    load_weights(enc_conv2_weight, weights_path_object/"enc_conv2_weight.bin", 32*32*3*3);
    load_weights(enc_conv2_bias, weights_path_object/"enc_conv2_bias.bin", 32);
    // load decoder weights
    load_weights(dec_tconv1_weight, weights_path_object/"dec_tconv1_weight.bin", 32*32*3*3);
    load_weights(dec_tconv1_bias, weights_path_object/"dec_tconv1_bias.bin", 32);
    load_weights(dec_tconv2_weight, weights_path_object/"dec_tconv2_weight.bin", 32*32*3*3);
    load_weights(dec_tconv2_bias, weights_path_object/"dec_tconv2_bias.bin", 32);
    load_weights(dec_conv1_weight, weights_path_object/"dec_conv1_weight.bin", 32*32*3*3);
    load_weights(dec_conv1_bias, weights_path_object/"dec_conv1_bias.bin", 1);

    for(int i=0; i<32; i++)
    {
        std::cout<< enc_conv1_bias[i] << std::endl;
    }
    
    // preprocess image: uint8 [0, 255] -> float [0, 1] 
    uint8_t *dev_uint_input;
    float *dev_input;
    cudaMalloc((void**)&dev_uint_input, height *width * sizeof(uint8_t));
    cudaMalloc((void**)&dev_input, height *width * sizeof(float));
    cudaMemcpy(dev_uint_input, input, height * width * sizeof(uint8_t), cudaMemcpyHostToDevice);
    dim3 img_block_size(32, 32);
    dim3 img_grid_size(width/32+1, height/32+1);
    img2float<<<img_grid_size, img_block_size>>>(dev_uint_input, dev_input, height, width);
    // preprocessing done

    //*****ENCODER*****
    // - layer 1
    int out_channels = 32;
    int in_channels = 1;
    float *dev_pool_layer1;
    int layer1_out_height = height/2, layer1_out_width = width/2;
    cudaMalloc((void**)&dev_pool_layer1, layer1_out_height * layer1_out_width * out_channels * sizeof(float));
    encoder_layer(dev_input, dev_pool_layer1, enc_conv1_weight, enc_conv1_bias, in_channels, out_channels, height, width);
    // - layer 2
    out_channels = 32;
    in_channels = 32;
    float *dev_pool_layer2;
    int layer2_out_height = layer1_out_height/2, layer2_out_width = layer1_out_width/2;
    cudaMalloc((void**)&dev_pool_layer2, layer2_out_height * layer2_out_width * out_channels * sizeof(float));
    encoder_layer(dev_pool_layer1, dev_pool_layer2, enc_conv2_weight, enc_conv2_bias, in_channels, out_channels, layer1_out_height, layer1_out_width);
    //*****DECODER*****
    // - layer 3
    out_channels = 32;
    in_channels = 32;
    float *dev_trans_layer3;
    int layer3_out_height = layer1_out_height, layer3_out_width = layer1_out_width;
    cudaMalloc((void**)&dev_trans_layer3, layer3_out_height * layer3_out_width * out_channels * sizeof(float));
    decoder_layer(dev_pool_layer2, dev_trans_layer3, dec_tconv1_weight, dec_tconv1_bias, in_channels, out_channels, layer2_out_height, layer2_out_width);
    // - layer 4
    out_channels = 32;
    in_channels = 32;
    float *dev_trans_layer4;
    int layer4_out_height = height, layer4_out_width = width;
    cudaMalloc((void**)&dev_trans_layer4, layer4_out_height * layer4_out_width * out_channels * sizeof(float));
    decoder_layer(dev_trans_layer3, dev_trans_layer4, dec_tconv2_weight, dec_tconv2_bias, in_channels, out_channels, layer3_out_height, layer3_out_width);
    // - layer 5
    out_channels = 1;
    in_channels = 32;
    float *dev_result;
    cudaMalloc((void**)&dev_result, height * width * out_channels * sizeof(float));
    refine_layer(dev_trans_layer4, dev_result, dec_conv1_weight, dec_conv1_bias, in_channels, out_channels, layer4_out_height, layer4_out_width);

    // float pool_result[1*28*28];
    // cudaMemcpy(pool_result, dev_result, 1*28*28 * sizeof(float), cudaMemcpyDeviceToHost);
    // print_array(pool_result, height, width, 1);
    



    img2uint<<<img_grid_size, img_block_size>>>(dev_result, dev_uint_input, height, width);
    cudaMemcpy(result, dev_uint_input, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // // begin experiments
    // float array[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27};
    // int array_h = 3;
    // int array_w = 3;
    // int array_c = 3;
    // int up_pad = 1;
    // int down_pad = 1;
    // int left_pad = 1;
    // int right_pad = 1;
    // float padded_array[(array_h + up_pad + down_pad) * (array_w + left_pad + right_pad) * array_c] = {0};
    // print_array(array, array_h, array_w, array_c);
    // //print_array(padded_array, array_h, array_w, array_c);

    // float *dev_array;
    // float *dev_padded_array;
    
    // cudaMalloc((void**)&dev_array, array_h * array_w * array_c * sizeof(float));
    // cudaMalloc((void**)&dev_padded_array, (array_h + up_pad + down_pad) * (array_w + left_pad + right_pad) * array_c * sizeof(float));

    // cudaMemcpy(dev_array, array, array_h * array_w * array_c * sizeof(float), cudaMemcpyHostToDevice);
    // const dim3 exp_block_size(32, 32, 1);
    // const dim3 exp_grid_size((array_w + left_pad + right_pad)/32+1, (array_h + up_pad + down_pad)/32+1, array_c);
    // ZeroPad2D<<<exp_grid_size, exp_block_size>>>(dev_array, dev_padded_array, array_c, array_h, array_w, up_pad, down_pad, left_pad, right_pad);
    // cudaMemcpy(padded_array, dev_padded_array, (array_h + up_pad + down_pad) * (array_w + left_pad + right_pad) * array_c * sizeof(float), cudaMemcpyDeviceToHost);
    // // convolution
    // int out_channels = 2;
    // float weight[] = {
    //     1,2,0,-1,1,3,2,1,0, 2,1,2,0,1,2,0,0,1, 1,1,0,2,1,2,0,1,0,
    //     1,1,2,0,3,0,1,2,3, -1,2,1,0,1,2,0,1,0, 1,1,0,2,1,0,2,2,2,
    // };
    // float bias[] = {1, 2};

    // float *dev_convolved_array;
    // float *dev_weight;
    // float *dev_bias;
    
    // cudaMalloc((void**)&dev_convolved_array, array_h * array_w * out_channels * sizeof(float));
    // cudaMalloc((void**)&dev_weight, 3 * 3 * 6 * sizeof(float));
    // cudaMalloc((void**)&dev_bias, 2 * sizeof(float));
    // cudaMemcpy(dev_weight, weight, 3 * 3 * 6 * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_bias, bias, 2 * sizeof(float), cudaMemcpyHostToDevice);
    // dim3 conv_block_size(32, 32, 1);
    // dim3 conv_grid_size(array_w/32+1, array_h/32+1, out_channels);
    // Conv2D<<<conv_grid_size, conv_block_size>>>(dev_padded_array, dev_weight, dev_bias, dev_convolved_array, 3, 2, array_h + up_pad + down_pad, array_w + left_pad + right_pad, 3, 3);
    // cudaMemcpy(array, dev_convolved_array, array_h * array_w * out_channels * sizeof(float), cudaMemcpyDeviceToHost);
    // // MaxPool2D
    // float *dev_pooled_array;
    // float pooled_array[2*2*3];
    // cudaMalloc((void**)&dev_pooled_array, ((array_w + left_pad + right_pad)/2) * ((array_h + up_pad + down_pad)/2) * array_c * sizeof(float));
    // dim3 pool_block_size(32, 32, 1);
    // dim3 pool_grid_size((array_w + left_pad + right_pad)/2/32+1, (array_h + up_pad + down_pad)/2/32+1, array_c);
    // MaxPool2D<<<pool_grid_size, pool_block_size>>>(dev_padded_array, dev_pooled_array, array_c, array_h + up_pad + down_pad, array_w + left_pad + right_pad, 2,2,2,2);
    // cudaMemcpy(pooled_array, dev_pooled_array, ((array_w + left_pad + right_pad)/2) * ((array_h + up_pad + down_pad)/2) * array_c * sizeof(float), cudaMemcpyDeviceToHost);
    // //Upsample
    // // MaxPool2D
    // float *dev_up_array;
    // float up_array[2*5*5];
    // cudaMalloc((void**)&dev_up_array, 5*5*2* sizeof(float));
    // dim3 up_block_size(32, 32, 1);
    // dim3 up_grid_size(5/32+1, 5/32+1, 2);
    // ChessUpsample2D<<<up_grid_size, up_block_size>>>(dev_convolved_array, dev_up_array, 2, 3, 3);
    // cudaMemcpy(up_array, dev_up_array, 2*5*5 * sizeof(float), cudaMemcpyDeviceToHost);
    // // Flip Weight
    // printf("####WEIGHT####\n");
    // print_array(weight, 3, 3, 3*2);
    // dim3 weight_block_size(32, 32, 1);
    // dim3 weight_grid_size(3/32+1, 3/32+1, 2);
    // FlipWeight2D<<<weight_grid_size, weight_block_size>>>(dev_weight, 2, 3, 3, 3);
    // cudaMemcpy(weight, dev_weight, 3*3*2*3 * sizeof(float), cudaMemcpyDeviceToHost);
    // printf("####F_WEIGHT####\n");
    // print_array(weight, 3, 3, 3*2);
    // //cudaFree(dev_array);
    // //cudaFree(dev_padded_array);
    
    // print_array(padded_array, array_h + up_pad + down_pad, array_w + left_pad + right_pad, array_c);
    // print_array(array, array_h, array_w, out_channels);
    // print_array(pooled_array, (array_h + up_pad + down_pad)/2, (array_w + left_pad + right_pad)/2, array_c);
    // print_array(up_array, 5, 5, 2);
    // // end experiment

    cudaFree(dev_input);
    cudaFree(dev_uint_input);
    
}

void encoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    float *dev_padded_layer1;
    int up_pad = 1, down_pad = 1, left_pad = 1, right_pad = 1;
    int padded_height= height+up_pad+down_pad, padded_width = width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded_layer1, padded_height * padded_width * in_channels * sizeof(float));
    dim3 enc1_pad_block_size(32, 32, 1);
    dim3 enc1_pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    ZeroPad2D<<<enc1_pad_grid_size, enc1_pad_block_size>>>(dev_input, dev_padded_layer1, in_channels, height, width, up_pad, down_pad, left_pad, right_pad);
    // -- convolve
    // ---- load weights
    float *dev_enc_conv1_weight;
    cudaMalloc((void**)&dev_enc_conv1_weight, out_channels*in_channels*3*3 * sizeof(float));
    cudaMemcpy(dev_enc_conv1_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice);
    float *dev_enc_conv1_bias;
    cudaMalloc((void**)&dev_enc_conv1_bias, out_channels * sizeof(float));
    cudaMemcpy(dev_enc_conv1_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);
    // ---- perform conv
    float *dev_conv_layer1;
    cudaMalloc((void**)&dev_conv_layer1, height * width * out_channels * sizeof(float));
    dim3 enc1_conv_block_size(32, 32, 1);
    dim3 enc1_conv_grid_size(width/32+1, height/32+1, out_channels);
    Conv2D<<<enc1_conv_grid_size, enc1_conv_block_size>>>
    (
        dev_padded_layer1, dev_enc_conv1_weight, dev_enc_conv1_bias, dev_conv_layer1,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    // relu
    ReLU<<<enc1_conv_grid_size, enc1_conv_block_size>>>(dev_conv_layer1, out_channels, height, width);
    // -- maxpool
    dim3 enc1_pool_block_size(32, 32, 1);
    dim3 enc1_pool_grid_size((width/2)/32+1, (height/2)/32+1, out_channels);
    MaxPool2D<<<enc1_pool_grid_size, enc1_pool_block_size>>>
    (
        dev_conv_layer1, dev_output, out_channels, height, width, 2,2,2,2
    );
}

void decoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    float *dev_upsampled;
    int up_height = height + height-1, up_width = width + width-1;
    cudaMalloc((void**)&dev_upsampled, up_height * up_width * in_channels * sizeof(float));
    dim3 up_block_size(32, 32, 1);
    dim3 up_grid_size(up_width/32+1, up_height/32+1, in_channels);
    ChessUpsample2D<<<up_grid_size, up_block_size>>>(dev_input, dev_upsampled, in_channels, height, width);


    float *dev_padded;
    int up_pad = 1, down_pad = 2, left_pad = 1, right_pad = 2;
    int padded_height= up_height+up_pad+down_pad, padded_width = up_width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    ZeroPad2D<<<pad_grid_size, pad_block_size>>>(dev_upsampled, dev_padded, in_channels, up_height, up_width, up_pad, down_pad, left_pad, right_pad);
    // -- convolve
    // ---- load weights
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    cudaMemcpy(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 weight_block_size(32, 32, 1);
    dim3 weight_grid_size(3/32+1, 3/32+1, out_channels);
    FlipWeight2D<<<weight_grid_size, weight_block_size>>>(dev_conv_weight, out_channels, in_channels, 3, 3);
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));
    cudaMemcpy(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // ---- perform conv
    int conv_height = height*2, conv_width = width*2;
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(conv_width/32+1, conv_height/32+1, out_channels);
    Conv2D<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight, dev_conv_bias, dev_output,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    // relu
    ReLU<<<conv_grid_size, conv_block_size>>>(dev_output, out_channels, conv_height, conv_width);
}

void refine_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    float *dev_padded;
    int up_pad = 1, down_pad = 1, left_pad = 1, right_pad = 1;
    int padded_height= height+up_pad+down_pad, padded_width = width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    ZeroPad2D<<<pad_grid_size, pad_block_size>>>(dev_input, dev_padded, in_channels, height, width, up_pad, down_pad, left_pad, right_pad);
    // -- convolve
    // ---- load weights
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    cudaMemcpy(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice);
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));
    cudaMemcpy(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);
    // ---- perform conv
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(width/32+1, height/32+1, out_channels);
    Conv2D<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight, dev_conv_bias, dev_output,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    Sigmoid<<<conv_grid_size, conv_block_size>>>(dev_output, out_channels, height, width);

}

void print_array(float * a, int h, int w, int c)
{
    std::cout << '\n';

    for(int k=0; k<c; k++)
    {
        for(int i=0; i<h; i++)
        {
            for(int j=0; j<w; j++)
            {
                std::cout << a[k*h*w + i*w + j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
}

void load_weights(float* weight_array, std::string weight_path, unsigned weight_size)
{
    std::ifstream ifs(weight_path, std::ifstream::binary);
    for(int i=0; i<weight_size; i++)
    {
        ifs.read(reinterpret_cast<char*>(&weight_array[i]), sizeof(float)); 
    }
    ifs.close();
}

__global__ void img2float(uint8_t *dev_input, float *dev_output, int height, int width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if(y > height-1 || x > width-1)
        return;
    
    dev_output[y*width + x] = float(dev_input[y*width + x]) / 255.;
}

__global__ void img2uint(float *dev_input, uint8_t *dev_output, int height, int width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    if(y > height-1 || x > width-1)
        return;
    
    float intensity = dev_input[y*width + x] * 255.;
    intensity = intensity > 255 ? 255 : intensity;
    intensity = intensity < 0 ? 0 : intensity;

    dev_output[y*width + x] = intensity;
}

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
    @param dev_output - output tensor
    @param in_channels, out_channels, in_height, in_width - input tensor shape
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

__global__ void ReLU(float *dev_input, int in_channels, int in_height, int in_width)
{
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (z > in_channels - 1 || y > in_height - 1 || x > in_width -1)
        return;
    
    float pixel = dev_input[z*in_height*in_width + y*in_width + x];
    dev_input[z*in_height*in_width + y*in_width + x] =  pixel > 0. ? pixel : 0.;
}

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