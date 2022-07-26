all: build
build:
	nvcc ./main.cu -arch=sm_86 `pkg-config opencv4 --cflags --libs` -o main.out
clean:
	rm -rf main.out