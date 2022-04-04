opencv4 = `pkg-config --cflags --libs opencv4`
cuda = -L/usr/local/cuda/lib64 -lcuda -lcudart

run: canny
	./canny.bin /home/v.pyatov/cuda_prac/Animation01_LDR_0000.png -sigma=3

canny: main.o canny_openmp.o canny_cuda.o
	g++ main.o canny_openmp.o canny_cuda.o -fopenmp -o canny.bin $(opencv4) $(cuda)

main.o: main.cpp canny_openmp/canny_openmp.hpp
	g++ -c main.cpp $(opencv4)

canny_openmp.o: canny_openmp/canny_openmp.cpp canny_openmp/canny_openmp.hpp
	g++ -c canny_openmp/canny_openmp.cpp -fopenmp $(opencv4)

canny_cuda.o: canny_cuda/canny_cuda.cu canny_cuda/canny_cuda.hpp
	nvcc -c canny_cuda/canny_cuda.cu
clean:
	rm main.o canny_openmp.o canny.bin