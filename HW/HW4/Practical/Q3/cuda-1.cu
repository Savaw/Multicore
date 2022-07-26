
#include <iostream>
#include <math.h>
#include <string>

#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;

__global__
void invert(uchar *image, uchar *inverted_image, size_t total_size) {
    for (int i = 0; i < total_size; i++)
        inverted_image[i] = 255 - image[i];
}


int main(void) {
    cudaEvent_t start, stop;
    float ms = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    string in_path = "pics/pic2.jpg";
    string out_path = "out.jpg";

    cout << "Reading image..." << endl;
    Mat M = imread(in_path);
    Size imgsize = M.size();

    Mat inverted_M(M.size(), M.type(), Scalar(0, 0, 0));

    size_t total_size = imgsize.width * imgsize.height * M.channels();
    int total_bytes = total_size * sizeof(uchar);

    uchar *image_host, *inverted_image_host;

    image_host = M.data;
    inverted_image_host = inverted_M.data;

    uchar *image, *inverted_image;

    cudaMalloc(&image, total_bytes);
    cudaMalloc(&inverted_image, total_bytes);

    cudaMemcpy(image, image_host, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(inverted_image, inverted_image_host, total_bytes, cudaMemcpyHostToDevice);

    cout << "inverting..." << endl;

    int blockSize = 1;
    int numBlocks = 1;

    cudaEventRecord(start);
    invert<<<numBlocks, blockSize>>>(image, inverted_image, total_size);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&ms, start, stop);
    cout << "Time: " << ms << "ms" << endl;

    cudaMallocHost(&inverted_image_host, total_bytes);

    cudaMemcpy(inverted_image_host, inverted_image, total_bytes, cudaMemcpyDeviceToHost);

    inverted_M = Mat(imgsize, M.type(), inverted_image_host);

    cout << "Writing image..." << endl;
    imwrite(out_path, inverted_M);

    cudaFree(image);
    cudaFree(inverted_image);

    cout << "DONE" << endl;
    return 0;
}
