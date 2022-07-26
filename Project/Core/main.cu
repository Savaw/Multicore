#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>

#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;


__global__
void adjust(uchar *input,
            uchar *output,
            float alpha,
            float beta,
            size_t size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    uchar pixel = input[idx];
    int value = alpha * pixel + beta;
    if (value > 255)
        value = 255;
    if (value < 0)
        value = 0;

    output[idx] = (uchar) value;

}

__global__
void conv(uchar *input,
          uchar *output,
          float *kernel_v,
          float *kernel_h,
          size_t rows,
          size_t cols,
          size_t kernel_size) {


    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int i = idx / cols;
    int j = idx % cols;

    float sum_v = 0, sum_h = 0;
    for (int kidx = 0; kidx < kernel_size * kernel_size; kidx++) {
        int ki = kidx / kernel_size;
        int kj = kidx % kernel_size;

        int new_i = i + ki - kernel_size / 2;
        int new_j = j + kj - kernel_size / 2;

        if (new_i < 0 || new_i >= rows || new_j < 0 || new_j >= cols)
            continue;

        int new_idx = new_i * cols + new_j;

        sum_v += kernel_v[kidx] * (float) input[new_idx];
        sum_h += kernel_h[kidx] * (float) input[new_idx];
    }

    int value = (int) sqrt(sum_v * sum_v + sum_h * sum_h);
    if (value > 255)
        value = 255;
    if (value < 0)
        value = 0;

    output[idx] = (uchar) value;
}

int main(int argc, char *argv[]) {
    cudaEvent_t start, stop;
    float ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const char *in_path = (const char *) argv[1];
    const char *alpha_arg = (const char *) argv[2];
    const char *beta_arg = (const char *) argv[3];

    float alpha = atof(alpha_arg);
    float beta = atof(beta_arg);

    cout << "alpha: " << alpha << " beta: " << beta << endl;

    string adjust_path = "sobel-out-adjust.jpg";
    string out_path = "sobel-out.jpg";

    cout << "Reading image..." << endl;
    Mat M = imread(in_path, IMREAD_GRAYSCALE);
    size_t img_height = M.size().height;
    size_t img_width = M.size().width;

    cout << "Image size: " << M.size() << "  channels: " << M.channels() << " type:" << M.type() << endl;
    Mat output_M = Mat(M.size(), M.type(), Scalar(0));
    Mat adjusted_M = Mat(M.size(), M.type(), Scalar(0));

    size_t kernel_size = 3;
    float kernel_v_data[] = {-1, 0, 1,
                             -2, 0, 2,
                             -1, 0, 1};

    float kernel_h_data[] = {1, 2, 1,
                             0, 0, 0,
                             -1, -2, -1};


    // Moving to device

    size_t total_size = img_height * img_width;
    size_t total_size_bytes = total_size * sizeof(uchar);

    uchar *intput_image_host = (uchar *) malloc(total_size_bytes);
    uchar *output_image_host = (uchar *) malloc(total_size_bytes);
    uchar *adjusted_image_host = (uchar *) malloc(total_size_bytes);

    intput_image_host = M.data;
    output_image_host = output_M.data;
    adjusted_image_host = adjusted_M.data;

    uchar *intput_image_device, *output_image_device, *adjusted_image_device;

    cudaMalloc(&intput_image_device, total_size_bytes);
    cudaMalloc(&output_image_device, total_size_bytes);
    cudaMalloc(&adjusted_image_device, total_size_bytes);

    cudaMemcpy(intput_image_device, intput_image_host, total_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(output_image_device, output_image_host, total_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(adjusted_image_device, adjusted_image_host, total_size_bytes, cudaMemcpyHostToDevice);

    float *kernel_v_device, *kernel_h_device;

    size_t kernel_size_bytes = kernel_size * kernel_size * sizeof(float);
    cudaMalloc(&kernel_v_device, kernel_size_bytes);
    cudaMalloc(&kernel_h_device, kernel_size_bytes);

    cudaMemcpy(kernel_v_device, kernel_v_data, kernel_size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_h_device, kernel_h_data, kernel_size_bytes, cudaMemcpyHostToDevice);


    int blockSize = 256;
    int numBlocks = (total_size + blockSize - 1) / blockSize;

    cout << "Adjusting..." << endl;
    adjust<<<numBlocks, blockSize>>>(intput_image_device,
                                     adjusted_image_device,
                                     alpha,
                                     beta,
                                     img_height * img_width);

    cudaDeviceSynchronize();
    cudaMallocHost(&adjusted_image_host, total_size_bytes);
    cudaMemcpy(adjusted_image_host, adjusted_image_device, total_size_bytes, cudaMemcpyDeviceToHost);

    adjusted_M = Mat(M.size(), M.type(), adjusted_image_host);

    cout << "Writing adjusted image..." << endl;
    imwrite(adjust_path, adjusted_M);

    cout << "Applying filter ..." << endl;
    cudaEventRecord(start);
    conv<<<numBlocks, blockSize>>>(adjusted_image_device, output_image_device,
                                   kernel_v_device, kernel_h_device,
                                   img_height, img_width,
                                   kernel_size);


    // Moving to host
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Time: " << ms << "ms" << endl;
    cudaMallocHost(&output_image_host, total_size_bytes);
    cudaMemcpy(output_image_host, output_image_device, total_size_bytes, cudaMemcpyDeviceToHost);

    output_M = Mat(M.size(), M.type(), output_image_host);

    cout << "Writing image..." << endl;
    imwrite(out_path, output_M);

    cout << "DONE" << endl;
    return 0;
}