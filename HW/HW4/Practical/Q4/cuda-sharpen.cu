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


__global__
void conv(byte *input, 
          byte *output, 
          float *kernel,
          size_t rows,
          size_t cols,
          int bytesPerPixel,
          size_t kernel_size)
{

  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int i = idx / cols;
  int j = idx % cols;

  for (int rgb = 0; rgb < 3; rgb ++)
  { 
    float sum = 0;
    for (int kidx = 0; kidx < kernel_size*kernel_size; kidx++)
      {
        int ki = kidx / kernel_size;
        int kj = kidx % kernel_size;

        int new_i = i + ki - kernel_size/2;
        int new_j = j + kj - kernel_size/2;

        if (new_i < 0 || new_i >= rows || new_j < 0 || new_j >= cols)
          continue;

        int new_idx = new_i * cols + new_j;

        sum += kernel[kidx] * (float)input[new_idx*bytesPerPixel + rgb];
      }

    int value = (int)sum;
    if (value > 255)
      value = 255;
    if (value < 0)
      value = 0;
    
    output[idx*bytesPerPixel + rgb] = (byte)value;
  }
}

int main()
{
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


  /* Move images to device */
  cudaMemcpy(pixels_top_device, pixels_top, total_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(pixels_bg_device, pixels_bg, total_size_bytes, cudaMemcpyHostToDevice);


  /* start replacing green screen */
  int blockSize = 256;
  int numBlocks = (total_size + blockSize - 1) / blockSize;
  replace<<<blockSize,numBlocks>>>(pixels_top_device, 
                  pixels_bg_device,
                  width, 
                  height, 
                  bytesPerPixel);

  cudaDeviceSynchronize();
  

  /* Define sharpening filter */
  float kernel_data[] = { 0,      -1.0/2,   0, 
                          -1.0/2, 3,        -1.0/2, 
                          0,      -1.0/2,   0};

  size_t kernel_size = 3;
  size_t kernel_size_bytes = kernel_size*kernel_size*sizeof(float);

  float *kernel_device;
  cudaMalloc(&kernel_device, kernel_size_bytes);

  cudaMemcpy(kernel_device, kernel_data, kernel_size_bytes, cudaMemcpyHostToDevice);

  /* start sharpening */
  conv<<<blockSize,numBlocks>>>(pixels_top_device, 
                                pixels_bg_device, 
                                kernel_device, 
                                height,
                                width,
                                bytesPerPixel,
                                kernel_size);

  cudaDeviceSynchronize();

  /* Move result to host */
  cudaMemcpy(pixels_bg, pixels_bg_device, total_size_bytes, cudaMemcpyDeviceToHost);
 
  /* write new image */
  WriteImage("sharpen.bmp", pixels_bg, width, height, bytesPerPixel);
  
  /* free everything */
  free(pixels_top);
  free(pixels_bg);
  cudaFree(pixels_top_device);
  cudaFree(pixels_bg_device);
  return 0;
}
