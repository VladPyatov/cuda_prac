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


std::pair<float, float> denoise(const uint8_t* input, uint8_t* result, param* weights, int height, int width)
{
    // start timer
	cudaEvent_t global_start, global_stop;
	cudaEventCreate(&global_start);
	cudaEventCreate(&global_stop); 
    float time = 0;
    cudaEventRecord(global_start, 0);

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
    int out_channels = weights[0].dim_0;
    int in_channels = weights[0].dim_1;
    float *dev_pool_layer1;
    int layer1_out_height = height/2, layer1_out_width = width/2;
    cudaMalloc((void**)&dev_pool_layer1, layer1_out_height * layer1_out_width * out_channels * sizeof(float));
    float layer1_time = encoder_layer(dev_input, dev_pool_layer1, weights[0].weight, weights[0].bias, in_channels, out_channels, height, width);
    // - layer 2
    out_channels = weights[1].dim_0;
    in_channels = weights[1].dim_1;
    float *dev_pool_layer2;
    int layer2_out_height = layer1_out_height/2, layer2_out_width = layer1_out_width/2;
    cudaMalloc((void**)&dev_pool_layer2, layer2_out_height * layer2_out_width * out_channels * sizeof(float));
    float layer2_time = encoder_layer(dev_pool_layer1, dev_pool_layer2, weights[1].weight, weights[1].bias, in_channels, out_channels, layer1_out_height, layer1_out_width);
    //*****DECODER*****
    // - layer 3
    out_channels = weights[2].dim_1;
    in_channels = weights[2].dim_0;
    float *dev_trans_layer3;
    int layer3_out_height = layer1_out_height, layer3_out_width = layer1_out_width;
    cudaMalloc((void**)&dev_trans_layer3, layer3_out_height * layer3_out_width * out_channels * sizeof(float));
    float layer3_time = decoder_layer(dev_pool_layer2, dev_trans_layer3, weights[2].weight, weights[2].bias, in_channels, out_channels, layer2_out_height, layer2_out_width);
    // - layer 4
    out_channels = weights[3].dim_1;
    in_channels = weights[3].dim_0;
    float *dev_trans_layer4;
    int layer4_out_height = height, layer4_out_width = width;
    cudaMalloc((void**)&dev_trans_layer4, layer4_out_height * layer4_out_width * out_channels * sizeof(float));
    float layer4_time = decoder_layer(dev_trans_layer3, dev_trans_layer4, weights[3].weight, weights[3].bias, in_channels, out_channels, layer3_out_height, layer3_out_width);
    // - layer 5
    out_channels = weights[4].dim_0;
    in_channels = weights[4].dim_1;
    float *dev_result;
    cudaMalloc((void**)&dev_result, height * width * out_channels * sizeof(float));
    float layer5_time = refine_layer(dev_trans_layer4, dev_result, weights[4].weight, weights[4].bias, in_channels, out_channels, layer4_out_height, layer4_out_width);
    // postprocessing
    img2uint<<<img_grid_size, img_block_size>>>(dev_result, dev_uint_input, height, width);
    cudaMemcpy(result, dev_uint_input, height * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(dev_input);
    cudaFree(dev_uint_input);
    cudaFree(dev_pool_layer1);
    cudaFree(dev_pool_layer2);
    cudaFree(dev_trans_layer3);
    cudaFree(dev_trans_layer4);
    cudaFree(dev_result);

    cudaEventRecord(global_stop, 0);
	cudaEventSynchronize(global_stop);
    cudaEventElapsedTime(&time, global_start, global_stop);
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_stop);
    
    return std::make_pair(time, layer1_time + layer2_time + layer3_time + layer4_time + layer5_time);
}


float encoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    // memory
    float *dev_padded;
    int up_pad = 1, down_pad = 1, left_pad = 1, right_pad = 1;
    int padded_height= height+up_pad+down_pad, padded_width = width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));
    float *dev_conv;
    cudaMalloc((void**)&dev_conv, height * width * out_channels * sizeof(float));

    // -- timer on
    // cudaEvent_t pad_start, pad_stop;
	// cudaEventCreate(&pad_start);
	// cudaEventCreate(&pad_stop); 
    // float pad_time = 0;
    // cudaEventRecord(pad_start, 0);
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    ZeroPad2D<<<pad_grid_size, pad_block_size, 0, stream[0]>>>(dev_input, dev_padded, in_channels, height, width, up_pad, down_pad, left_pad, right_pad);
    // -- timer off
    // cudaDeviceSynchronize();
    // cudaEventRecord(pad_stop, 0);
	// cudaEventSynchronize(pad_stop);
    // cudaEventElapsedTime(&pad_time, pad_start, pad_stop);
    
    // - convolution
    // -- load weights
    cudaMemcpyAsync(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    // -- timer on
    // cudaEvent_t conv_start, conv_stop;
	// cudaEventCreate(&conv_start);
	// cudaEventCreate(&conv_stop); 
    // float conv_time = 0;
    // cudaEventRecord(conv_start, 0);
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(width/32+1, height/32+1, out_channels);
    SharedConv2DReLU<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight, dev_conv_bias, dev_conv,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    // -- relu
    //ReLU<<<conv_grid_size, conv_block_size>>>(dev_conv, out_channels, height, width);
    // -- timer off
    // cudaDeviceSynchronize();
    // cudaEventRecord(conv_stop, 0);
	// cudaEventSynchronize(conv_stop);
    // cudaEventElapsedTime(&conv_time, conv_start, conv_stop);

    // - maxpool
    // -- timer on
    // cudaEvent_t pool_start, pool_stop;
	// cudaEventCreate(&pool_start);
	// cudaEventCreate(&pool_stop); 
    // float pool_time = 0;
    // cudaEventRecord(pool_start, 0);
    dim3 pool_block_size(32, 32, 1);
    dim3 pool_grid_size((width/2)/32+1, (height/2)/32+1, out_channels);
    MaxPool2D<<<pool_grid_size, pool_block_size>>>
    (
        dev_conv, dev_output, out_channels, height, width, 2,2,2,2
    );
    // -- timer off
    // cudaDeviceSynchronize();
    // cudaEventRecord(pool_stop, 0);
	// cudaEventSynchronize(pool_stop);
    // cudaEventElapsedTime(&pool_time, pool_start, pool_stop);

    // cudaEventDestroy(pad_start);
    // cudaEventDestroy(pad_stop);
    // cudaEventDestroy(conv_start);
    // cudaEventDestroy(conv_stop);
    // cudaEventDestroy(pool_start);
    // cudaEventDestroy(pool_stop);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaFree(dev_padded);
    cudaFree(dev_conv_weight);
    cudaFree(dev_conv_bias);
    cudaFree(dev_conv);

    //return pad_time + conv_time + pool_time;
    return 0;
}


float decoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    // memory
    float *dev_upsampled;
    int up_height = height + height-1, up_width = width + width-1;
    cudaMalloc((void**)&dev_upsampled, up_height * up_width * in_channels * sizeof(float));
    //
    float *dev_padded;
    int up_pad = 1, down_pad = 2, left_pad = 1, right_pad = 2;
    int padded_height= up_height+up_pad+down_pad, padded_width = up_width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    //
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    //
    float *dev_conv_weight_flip;
    cudaMalloc((void**)&dev_conv_weight_flip, out_channels*in_channels*3*3 * sizeof(float));
    //
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));

    // cudaEvent_t up_start, up_stop;
	// cudaEventCreate(&up_start);
	// cudaEventCreate(&up_stop); 
    // float up_time = 0;
    // cudaEventRecord(up_start, 0);
    dim3 up_block_size(32, 32, 1);
    dim3 up_grid_size(up_width/32+1, up_height/32+1, in_channels);
    ChessUpsample2D<<<up_grid_size, up_block_size,0, stream[0]>>>(dev_input, dev_upsampled, in_channels, height, width);
    // cudaDeviceSynchronize();
    // cudaEventRecord(up_stop, 0);
	// cudaEventSynchronize(up_stop);
    // cudaEventElapsedTime(&up_time, up_start, up_stop);

    // cudaEvent_t pad_start, pad_stop;
	// cudaEventCreate(&pad_start);
	// cudaEventCreate(&pad_stop); 
    // float pad_time = 0;
    // cudaEventRecord(pad_start, 0);
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    ZeroPad2D<<<pad_grid_size, pad_block_size,0, stream[0]>>>(dev_upsampled, dev_padded, in_channels, up_height, up_width, up_pad, down_pad, left_pad, right_pad);
    // cudaDeviceSynchronize();
    // cudaEventRecord(pad_stop, 0);
	// cudaEventSynchronize(pad_stop);
    // cudaEventElapsedTime(&pad_time, pad_start, pad_stop);
    // -- convolve
    // ---- load weights
    
    cudaMemcpyAsync(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    
    // cudaEvent_t trans_start, trans_stop;
	// cudaEventCreate(&trans_start);
	// cudaEventCreate(&trans_stop); 
    // float trans_time = 0;
    // cudaEventRecord(trans_start, 0);
    dim3 weight_block_size(32, 32, 1);
    dim3 weight_grid_size(3/32+1, 3/32+1, out_channels*in_channels);
    TransposeKernel<<<weight_grid_size, weight_block_size, 0, stream[1]>>>(dev_conv_weight, dev_conv_weight_flip, out_channels, in_channels, 3, 3);
    // cudaDeviceSynchronize();
    // cudaEventRecord(trans_stop, 0);
	// cudaEventSynchronize(trans_stop);
    // cudaEventElapsedTime(&trans_time, trans_start, trans_stop);
    // --- load bias

    // ---- perform conv
    // cudaEvent_t conv_start, conv_stop;
	// cudaEventCreate(&conv_start);
	// cudaEventCreate(&conv_stop); 
    // float conv_time = 0;
    // cudaEventRecord(conv_start, 0);
    int conv_height = height*2, conv_width = width*2;
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(conv_width/32+1, conv_height/32+1, out_channels);
    SharedConv2DReLU<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight_flip, dev_conv_bias, dev_output,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    // relu
    //ReLU<<<conv_grid_size, conv_block_size>>>(dev_output, out_channels, conv_height, conv_width);
    // cudaDeviceSynchronize();
    // cudaEventRecord(conv_stop, 0);
	// cudaEventSynchronize(conv_stop);
    // cudaEventElapsedTime(&conv_time, conv_start, conv_stop);

    // cudaEventDestroy(up_start);
    // cudaEventDestroy(up_stop);
    // cudaEventDestroy(pad_start);
    // cudaEventDestroy(pad_stop);
    // cudaEventDestroy(trans_start);
    // cudaEventDestroy(trans_stop);
    // cudaEventDestroy(conv_start);
    // cudaEventDestroy(conv_stop);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    cudaFree(dev_upsampled);
    cudaFree(dev_padded);
    cudaFree(dev_conv_weight);
    cudaFree(dev_conv_bias);
    cudaFree(dev_conv_weight_flip);

    //return up_time + pad_time + trans_time + conv_time;
    return 0;
}


float refine_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    float *dev_padded;
    int up_pad = 1, down_pad = 1, left_pad = 1, right_pad = 1;
    int padded_height= height+up_pad+down_pad, padded_width = width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));

    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    // cudaEvent_t pad_start, pad_stop;
	// cudaEventCreate(&pad_start);
	// cudaEventCreate(&pad_stop); 
    // float pad_time = 0;
    // cudaEventRecord(pad_start, 0);
    ZeroPad2D<<<pad_grid_size, pad_block_size, 0, stream[0]>>>(dev_input, dev_padded, in_channels, height, width, up_pad, down_pad, left_pad, right_pad);
    // cudaStreamSynchronize(stream[0]);
    // cudaEventRecord(pad_stop, 0);
	// cudaEventSynchronize(pad_stop);
    // cudaEventElapsedTime(&pad_time, pad_start, pad_stop);

    // -- convolve
    // ---- load weights
    cudaMemcpyAsync(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice, stream[1]);
    // ---- perform conv
    // cudaEvent_t conv_start, conv_stop;
	// cudaEventCreate(&conv_start);
	// cudaEventCreate(&conv_stop); 
    // float conv_time = 0;
    // cudaEventRecord(conv_start, 0);
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(width/32+1, height/32+1, out_channels);
    SharedConv2DSigmoid<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight, dev_conv_bias, dev_output,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    //Sigmoid<<<conv_grid_size, conv_block_size>>>(dev_output, out_channels, height, width);
    // cudaStreamSynchronize(stream[1]);
    // cudaEventRecord(conv_stop, 0);
	// cudaEventSynchronize(conv_stop);
    // cudaEventElapsedTime(&conv_time, conv_start, conv_stop);

    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    // cudaEventDestroy(conv_start);
    // cudaEventDestroy(conv_stop);
    // cudaEventDestroy(pad_start);
    // cudaEventDestroy(pad_stop);
    cudaFree(dev_padded);
    cudaFree(dev_conv_weight);
    cudaFree(dev_conv_bias);

    //return conv_time+pad_time;
    return 0;
}