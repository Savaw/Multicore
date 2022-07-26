all: build
cuda1:
	nvcc -arch=sm_86 ./cuda-1.cu  -o cuda-1.out
cuda2:
	nvcc -arch=sm_86 ./cuda-2.cu  -o cuda-2.out
cuda-sharpen:
	nvcc -arch=sm_86 ./cuda-sharpen.cu  -o cuda-sharpen.out
build: cuda1 cuda2 cuda-sharpen
clean:
	rm -rf main.out