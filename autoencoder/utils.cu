#include "utils.cuh"


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