opencv4 = `pkg-config --cflags --libs opencv4`

run: canny
	./canny.bin /home/v.pyatov/cuda_prac/Animation01_LDR_0000.png -sigma=5

canny: main.o canny_openmp.o
	g++ main.o canny_openmp.o -fopenmp -o canny.bin $(opencv4)

main.o: main.cpp canny_openmp/canny_openmp.hpp
	g++ -c main.cpp -fopenmp $(opencv4)

canny_openmp.o: canny_openmp/canny_openmp.cpp canny_openmp/canny_openmp.hpp
	g++ -c canny_openmp/canny_openmp.cpp -fopenmp $(opencv4)

clean:
	rm main.o canny_openmp.o canny.bin