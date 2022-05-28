opencv4 = `pkg-config --cflags --libs opencv4`
cuda = -L/usr/local/cuda/lib64 -lcuda -lcudart

denoise: main.o cnn_autoencoder.o layers.o shared_layers.o utils.o
	g++ main.o cnn_autoencoder.o layers.o shared_layers.o utils.o -fopenmp -o denoiser.bin $(opencv4) $(cuda)

main.o: main.cpp
	g++ -std=c++17 -c main.cpp $(opencv4)

cnn_autoencoder.o: cnn_autoencoder.cu cnn_autoencoder.cuh
	nvcc -std=c++17 -c cnn_autoencoder.cu

layers.o: layers.cu layers.cuh
	nvcc -std=c++17 -c layers.cu

shared_layers.o: shared_layers.cu shared_layers.cuh
	nvcc -std=c++17 -c shared_layers.cu

utils.o: utils.cu utils.cuh
	nvcc -std=c++17 -c utils.cu

clean:
	rm main.o cnn_autoencoder.o layers.o  utils.o denoiser.bin