all: build
cuda1:
	nvcc -arch=sm_86 ./cuda-1.cu `pkg-config opencv4 --cflags --libs` -o cuda-1.out
cuda2:
	nvcc -arch=sm_86 ./cuda-2.cu `pkg-config opencv4 --cflags --libs` -o cuda-2.out
cpu:
	g++ cpu.cpp `pkg-config opencv4 --cflags --libs` -o cpu.out
build: cuda1 cuda2 cpu
clean:
	rm -rf main.out