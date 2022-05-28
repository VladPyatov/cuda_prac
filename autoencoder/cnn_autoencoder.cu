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
    return std::make_pair(time, layer1_time + layer2_time + layer3_time + layer4_time + layer5_time);
}


float encoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    // cudaStream_t s1, s2;
    // cudaStreaCreate(&s1);
    // cudaStreamCreate(&s2);
    // kernel1<<<...,s1>>>(...);
    // kernel2<<<...,s2>>>(...);
    // - padding
    float *dev_padded;
    int up_pad = 1, down_pad = 1, left_pad = 1, right_pad = 1;
    int padded_height= height+up_pad+down_pad, padded_width = width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    // -- timer on
    cudaEvent_t pad_start, pad_stop;
	cudaEventCreate(&pad_start);
	cudaEventCreate(&pad_stop); 
    float pad_time = 0;
    cudaEventRecord(pad_start, 0);
    ZeroPad2D<<<pad_grid_size, pad_block_size>>>(dev_input, dev_padded, in_channels, height, width, up_pad, down_pad, left_pad, right_pad);
    // -- timer off
    cudaDeviceSynchronize();
    cudaEventRecord(pad_stop, 0);
	cudaEventSynchronize(pad_stop);
    cudaEventElapsedTime(&pad_time, pad_start, pad_stop);
    // - convolution
    // -- load weights
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    cudaMemcpy(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice);
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));
    cudaMemcpy(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);
    // -- perform conv
    float *dev_conv;
    cudaMalloc((void**)&dev_conv, height * width * out_channels * sizeof(float));
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(width/32+1, height/32+1, out_channels);
    // -- timer on
    cudaEvent_t conv_start, conv_stop;
	cudaEventCreate(&conv_start);
	cudaEventCreate(&conv_stop); 
    float conv_time = 0;
    cudaEventRecord(conv_start, 0);
    SharedConv2DReLU<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight, dev_conv_bias, dev_conv,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    // -- relu
    //ReLU<<<conv_grid_size, conv_block_size>>>(dev_conv, out_channels, height, width);
    // -- timer off
    cudaDeviceSynchronize();
    cudaEventRecord(conv_stop, 0);
	cudaEventSynchronize(conv_stop);
    cudaEventElapsedTime(&conv_time, conv_start, conv_stop);
    // - maxpool
    dim3 pool_block_size(32, 32, 1);
    dim3 pool_grid_size((width/2)/32+1, (height/2)/32+1, out_channels);
    // -- timer on
    cudaEvent_t pool_start, pool_stop;
	cudaEventCreate(&pool_start);
	cudaEventCreate(&pool_stop); 
    float pool_time = 0;
    cudaEventRecord(pool_start, 0);
    MaxPool2D<<<pool_grid_size, pool_block_size>>>
    (
        dev_conv, dev_output, out_channels, height, width, 2,2,2,2
    );
    // -- timer off
    cudaDeviceSynchronize();
    cudaEventRecord(pool_stop, 0);
	cudaEventSynchronize(pool_stop);
    cudaEventElapsedTime(&pool_time, pool_start, pool_stop);

    cudaEventDestroy(pad_start);
    cudaEventDestroy(pad_stop);
    cudaEventDestroy(conv_start);
    cudaEventDestroy(conv_stop);
    cudaEventDestroy(pool_start);
    cudaEventDestroy(pool_stop);
    cudaFree(dev_padded);
    cudaFree(dev_conv_weight);
    cudaFree(dev_conv_bias);
    cudaFree(dev_conv);

    return pad_time + conv_time + pool_time;
}


float decoder_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    float *dev_upsampled;
    int up_height = height + height-1, up_width = width + width-1;
    cudaMalloc((void**)&dev_upsampled, up_height * up_width * in_channels * sizeof(float));
    dim3 up_block_size(32, 32, 1);
    dim3 up_grid_size(up_width/32+1, up_height/32+1, in_channels);
    cudaEvent_t up_start, up_stop;
	cudaEventCreate(&up_start);
	cudaEventCreate(&up_stop); 
    float up_time = 0;
    cudaEventRecord(up_start, 0);
    ChessUpsample2D<<<up_grid_size, up_block_size>>>(dev_input, dev_upsampled, in_channels, height, width);
    cudaDeviceSynchronize();
    cudaEventRecord(up_stop, 0);
	cudaEventSynchronize(up_stop);
    cudaEventElapsedTime(&up_time, up_start, up_stop);

    float *dev_padded;
    int up_pad = 1, down_pad = 2, left_pad = 1, right_pad = 2;
    int padded_height= up_height+up_pad+down_pad, padded_width = up_width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    cudaEvent_t pad_start, pad_stop;
	cudaEventCreate(&pad_start);
	cudaEventCreate(&pad_stop); 
    float pad_time = 0;
    cudaEventRecord(pad_start, 0);
    ZeroPad2D<<<pad_grid_size, pad_block_size>>>(dev_upsampled, dev_padded, in_channels, up_height, up_width, up_pad, down_pad, left_pad, right_pad);
    cudaDeviceSynchronize();
    cudaEventRecord(pad_stop, 0);
	cudaEventSynchronize(pad_stop);
    cudaEventElapsedTime(&pad_time, pad_start, pad_stop);
    // -- convolve
    // ---- load weights
    float *dev_conv_weight;
    cudaMalloc((void**)&dev_conv_weight, out_channels*in_channels*3*3 * sizeof(float));
    cudaMemcpy(dev_conv_weight, weight, out_channels*in_channels*3*3 * sizeof(float), cudaMemcpyHostToDevice);
    dim3 weight_block_size(32, 32, 1);
    dim3 weight_grid_size(3/32+1, 3/32+1, out_channels*in_channels);
    float *dev_conv_weight_flip;
    cudaMalloc((void**)&dev_conv_weight_flip, out_channels*in_channels*3*3 * sizeof(float));
    cudaEvent_t trans_start, trans_stop;
	cudaEventCreate(&trans_start);
	cudaEventCreate(&trans_stop); 
    float trans_time = 0;
    cudaEventRecord(trans_start, 0);
    TransposeKernel<<<weight_grid_size, weight_block_size>>>(dev_conv_weight, dev_conv_weight_flip, out_channels, in_channels, 3, 3);
    cudaDeviceSynchronize();
    cudaEventRecord(trans_stop, 0);
	cudaEventSynchronize(trans_stop);
    cudaEventElapsedTime(&trans_time, trans_start, trans_stop);
    // --- load bias
    float *dev_conv_bias;
    cudaMalloc((void**)&dev_conv_bias, out_channels * sizeof(float));
    cudaMemcpy(dev_conv_bias, bias, out_channels * sizeof(float), cudaMemcpyHostToDevice);

    // ---- perform conv
    int conv_height = height*2, conv_width = width*2;
    dim3 conv_block_size(32, 32, 1);
    dim3 conv_grid_size(conv_width/32+1, conv_height/32+1, out_channels);
    cudaEvent_t conv_start, conv_stop;
	cudaEventCreate(&conv_start);
	cudaEventCreate(&conv_stop); 
    float conv_time = 0;
    cudaEventRecord(conv_start, 0);
    SharedConv2DReLU<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight_flip, dev_conv_bias, dev_output,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    // relu
    //ReLU<<<conv_grid_size, conv_block_size>>>(dev_output, out_channels, conv_height, conv_width);
    cudaDeviceSynchronize();
    cudaEventRecord(conv_stop, 0);
	cudaEventSynchronize(conv_stop);
    cudaEventElapsedTime(&conv_time, conv_start, conv_stop);

    cudaEventDestroy(up_start);
    cudaEventDestroy(up_stop);
    cudaEventDestroy(pad_start);
    cudaEventDestroy(pad_stop);
    cudaEventDestroy(trans_start);
    cudaEventDestroy(trans_stop);
    cudaEventDestroy(conv_start);
    cudaEventDestroy(conv_stop);
    cudaFree(dev_upsampled);
    cudaFree(dev_padded);
    cudaFree(dev_conv_weight);
    cudaFree(dev_conv_bias);
    cudaFree(dev_conv_weight_flip);

    return up_time + pad_time + trans_time + conv_time;
}


float refine_layer(float *dev_input, float *dev_output, float *weight, float *bias, int in_channels, int out_channels, int height, int width)
{
    float *dev_padded;
    int up_pad = 1, down_pad = 1, left_pad = 1, right_pad = 1;
    int padded_height= height+up_pad+down_pad, padded_width = width+left_pad+right_pad;
    cudaMalloc((void**)&dev_padded, padded_height * padded_width * in_channels * sizeof(float));
    dim3 pad_block_size(32, 32, 1);
    dim3 pad_grid_size(padded_width/32+1, padded_height/32+1, in_channels);
    cudaEvent_t pad_start, pad_stop;
	cudaEventCreate(&pad_start);
	cudaEventCreate(&pad_stop); 
    float pad_time = 0;
    cudaEventRecord(pad_start, 0);
    ZeroPad2D<<<pad_grid_size, pad_block_size>>>(dev_input, dev_padded, in_channels, height, width, up_pad, down_pad, left_pad, right_pad);
    cudaDeviceSynchronize();
    cudaEventRecord(pad_stop, 0);
	cudaEventSynchronize(pad_stop);
    cudaEventElapsedTime(&pad_time, pad_start, pad_stop);

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
    cudaEvent_t conv_start, conv_stop;
	cudaEventCreate(&conv_start);
	cudaEventCreate(&conv_stop); 
    float conv_time = 0;
    cudaEventRecord(conv_start, 0);
    SharedConv2DSigmoid<<<conv_grid_size, conv_block_size>>>
    (
        dev_padded, dev_conv_weight, dev_conv_bias, dev_output,
        in_channels, out_channels, padded_height, padded_width, 3, 3
    );
    //Sigmoid<<<conv_grid_size, conv_block_size>>>(dev_output, out_channels, height, width);
    cudaDeviceSynchronize();
    cudaEventRecord(conv_stop, 0);
	cudaEventSynchronize(conv_stop);
    cudaEventElapsedTime(&conv_time, conv_start, conv_stop);

    cudaEventDestroy(conv_start);
    cudaEventDestroy(conv_stop);
    cudaEventDestroy(pad_start);
    cudaEventDestroy(pad_stop);
    cudaFree(dev_padded);
    cudaFree(dev_conv_weight);
    cudaFree(dev_conv_bias);

    return conv_time+pad_time;
}