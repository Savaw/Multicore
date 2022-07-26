#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bmp.c"
#include <assert.h>

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

typedef unsigned int int32;
typedef short int16;
typedef unsigned char byte;

__device__
int is_green(byte r, byte g, byte b) {
    int green_dist = (byte) 255 + r + b,
            red_dist = (byte) 255 + g + b,
            blue_dist = (byte) 255 + r + g;
    return (green_dist < red_dist && green_dist < blue_dist);
}

__global__
void replace(byte *pixels_top,
             byte *pixels_bg,
             int width,
             int height,
             int bytesPerPixel) {

    int start = blockDim.x * blockIdx.x + threadIdx.x;
    if (is_green(pixels_top[start * bytesPerPixel],
                 pixels_top[start * bytesPerPixel + 1],
                 pixels_top[start * bytesPerPixel + 2])) {
        pixels_top[start * bytesPerPixel] = pixels_bg[start * bytesPerPixel];
        pixels_top[start * bytesPerPixel + 1] = pixels_bg[start * bytesPerPixel + 1];
        pixels_top[start * bytesPerPixel + 2] = pixels_bg[start * bytesPerPixel + 2];
    }
}

int main() {
    cudaEvent_t start, stop;
    float ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    /* start reading the file and its information*/
    byte *pixels_top, *pixels_bg;
    int32 width_top, width_bg;
    int32 height_top, height_bg;
    int32 bytesPerPixel_top, bytesPerPixel_bg;
    ReadImage("dino.bmp", &pixels_top, &width_top, &height_top, &bytesPerPixel_top);
    ReadImage("parking.bmp", &pixels_bg, &width_bg, &height_bg, &bytesPerPixel_bg);

    /* images should have color and be of the same size */
    assert(bytesPerPixel_top == 3);
    assert(width_top == width_bg);
    assert(height_top == height_bg);
    assert(bytesPerPixel_top == bytesPerPixel_bg);

    /* we can now work with one size */
    int32 width = width_top, height = height_top, bytesPerPixel = bytesPerPixel_top;

    size_t total_size = width * height;
    size_t total_size_bytes = total_size * bytesPerPixel;


    /* Malloc memory on device */
    byte *pixels_top_device, *pixels_bg_device;

    cudaMalloc(&pixels_top_device, total_size_bytes);
    cudaMalloc(&pixels_bg_device, total_size_bytes);

    cudaMemcpy(pixels_top_device, pixels_top, total_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(pixels_bg_device, pixels_bg, total_size_bytes, cudaMemcpyHostToDevice);


    /* start replacing green screen */
    int blockSize = 256;
    int numBlocks = (total_size + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    replace<<<blockSize, numBlocks>>>(pixels_top_device,
                                      pixels_bg_device,
                                      width,
                                      height,
                                      bytesPerPixel);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&ms, start, stop);
    printf("Time: %f ms\n", ms);

    cudaMallocHost(&pixels_top, total_size_bytes);
    /* Move images to host */
    cudaMemcpy(pixels_top, pixels_top_device, total_size_bytes, cudaMemcpyDeviceToHost);

    /* write new image */
    WriteImage("replaced.bmp", pixels_top, width, height, bytesPerPixel);

    /* free everything */
    cudaFree(pixels_top);
    free(pixels_bg);
    cudaFree(pixels_top_device);
    cudaFree(pixels_bg_device);
    return 0;
}
