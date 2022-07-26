all: build
cuda:
	nvcc ./main.cu -arch=sm_86 `pkg-config opencv4 --cflags --libs` -o main.out
cpu:
	g++ cpu.cpp `pkg-config opencv4 --cflags --libs` -o cpu.out
build: cuda cpu
clean:
	rm -rf main.out
	rm -rf cpu.out