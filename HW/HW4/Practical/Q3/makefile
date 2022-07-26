all:
	build
build:
	nvcc -arch=sm_86 ./cuda-1.cu `pkg-config opencv4 --cflags --libs` -o cuda-1.out
	nvcc -arch=sm_86 ./cuda-2.cu `pkg-config opencv4 --cflags --libs` -o cuda-2.out
	g++ cpu.cpp `pkg-config opencv4 --cflags --libs` -o cpu.out
clean:
	rm -rf main.out