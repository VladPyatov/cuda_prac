#include <filesystem>
#include <cstdio>
#include <iostream>
#include <regex>
#include <opencv2/imgcodecs.hpp> // imread, imwrite
#include <opencv2/core/utility.hpp> // parser
#include "cnn_autoencoder.cuh"
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
        "{help h usage ? |      | print this message     }"
        "{@image         |      | image for process      }"
        "{@weights       |      | CNN weights directory  }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    // parse parameters
    std::string image_path = parser.get<std::string>("@image");
    std::string weights_path = parser.get<std::string>("@weights");
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
    // denoise image
    denoise(input_img.data, result_img.data, weights_path, heigh, width);
    // save result
    std::string ext = fs::path(image_path).extension();
    image_path = std::regex_replace(image_path, std::regex(ext), "_denoised"+ext);
    cv::imwrite(image_path, result_img);

    
}