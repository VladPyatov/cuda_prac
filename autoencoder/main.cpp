#include <filesystem>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <regex>
#include <opencv2/imgcodecs.hpp> // imread, imwrite
#include <opencv2/core/utility.hpp> // parser
#include "cnn_autoencoder.cuh"
#include "omp.h"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    printf(
        "\n"
        "#################################################\n"
        "# Convolutional AutoEncoder. GPU implementation #\n"
        "#################################################\n"
        "\n"
        );
    
    // setup parser
    const std::string keys =
        "{help h usage ? |                     | print this message     }"
        "{@image         |                     | image for process      }"
        "{weights        |./autoencoder_weights| CNN weights directory  }"
        "{benchmark      |          0          | benhmark option        }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    // parse parameters
    std::string image_path = parser.get<std::string>("@image");
    std::string weights_path = parser.get<std::string>("weights");
    int benchmark = parser.get<int>("benchmark");
    cv:: Mat input_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(input_img.empty())
    {
        printf("Could not read the image: %s\n", image_path.c_str());
        return 1;
    }
    int width = input_img.cols;
    int heigh = input_img.rows;
    printf("Successfully loaded Image of size [%d, %d]: %s\n", heigh, width, image_path.c_str());

    // prepare output image
    cv::Mat result_img(heigh, width, CV_8UC1);

    // load weights
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
    load_weights(dec_conv1_weight, weights_path_object/"dec_conv1_weight.bin", 1*32*3*3);
    load_weights(dec_conv1_bias, weights_path_object/"dec_conv1_bias.bin", 1);
    // create array of parameters
    param weights_and_biases[] = 
    {
        param(enc_conv1_weight, enc_conv1_bias, 32, 1, 3 ,3),
        param(enc_conv2_weight, enc_conv2_bias, 32, 32, 3 ,3),
        param(dec_tconv1_weight, dec_tconv1_bias, 32, 32, 3 ,3),
        param(dec_tconv2_weight, dec_tconv2_bias, 32, 32, 3 ,3),
        param(dec_conv1_weight, dec_conv1_bias, 1, 32, 3 ,3),
    };
    // run benchmark
    if (benchmark > 0)
    {
        double start_time = omp_get_wtime();
        for(int i=0; i<benchmark; i++)
        {
            denoise(input_img.data, result_img.data, weights_and_biases, heigh, width);
        }
        double end_time = omp_get_wtime();
        printf("Mean time: %lf ms\n\n", 1000*(end_time - start_time)/double(benchmark));
    }
    else
    {
        // denoise image
        denoise(input_img.data, result_img.data, weights_and_biases, heigh, width);
        // save result
        std::string ext = fs::path(image_path).extension();
        image_path = std::regex_replace(image_path, std::regex(ext), "_denoised"+ext);
        cv::imwrite(image_path, result_img);

        std::cout << "Denoising results: " << image_path << '\n';
    }
}