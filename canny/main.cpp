#include <opencv2/imgcodecs.hpp> // imread, imwrite
#include <opencv2/core/utility.hpp> // parser
#include <cstdio>
#include "canny_openmp/canny_openmp.hpp"
#include "canny_cuda/canny_cuda.hpp"
#include "omp.h"
#include <iostream>
int main(int argc, char** argv)
{
    printf(
        "\n"
        "#####################################################\n"
        "# Canny Edge Detection. CPU and GPU implementations #\n"
        "#####################################################\n"
        "\n"
        );
    
    // setup parser
    const std::string keys =
        "{help h usage ? |      | print this message     }"
        "{@image         |      | image for process      }"
        "{sigma          | 3    | gaussian blur parameter}"
        "{low_t          | 0.05 | low threshold value    }"
        "{high_t         | 0.09 | high threshold value   }"
        ;
    cv::CommandLineParser parser(argc, argv, keys);

    // parse parameters
    std::string image_path = parser.get<std::string>("@image");
    cv:: Mat input_img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if(input_img.empty())
    {
        printf("Could not read the image: %s\n", image_path.c_str());
        return 1;
    }
    int width = input_img.cols;
    int heigh = input_img.rows;
    printf("Image of size [%d, %d]: %s\n", heigh, width, image_path.c_str());

    int sigma = parser.get<int>("sigma");
    float low_t = parser.get<float>("low_t");
    float high_t = parser.get<float>("high_t");
    printf("sigma=%d, low_threshold=%f, high_threshold=%f\n\n", sigma, low_t, high_t);

    // prepare output image
    cv::Mat result_img(heigh, width, CV_8UC1);

    // measure performance with 1 thread
    double start_time, end_time;
    omp_set_num_threads(1);
    start_time = omp_get_wtime();
    canny_openmp(input_img.data, result_img.data, heigh, width, low_t, high_t, sigma);
    end_time = omp_get_wtime();
    printf("CPU 1 thread time: %lf ms\n", 1000*(end_time - start_time));
    
    // measure performance with 8 threads
    omp_set_num_threads(8);
    start_time = omp_get_wtime();
    canny_openmp(input_img.data, result_img.data, heigh, width, low_t, high_t, sigma);
    end_time = omp_get_wtime();
    printf("CPU 8 threads time: %lf ms\n\n", 1000*(end_time - start_time));

    cv::imwrite("out_cpu.png", result_img);
    
    // measure performance on GPU
    canny_cuda(input_img.data, result_img.data, heigh, width, low_t, high_t, sigma);
    cv::imwrite("out_gpu.png", result_img);
    return 0;
}