#include <iostream>
#include <math.h>
#include <string>
#include <stdlib.h>
#include <chrono>

#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;


void adjust(Mat M, Mat output_M, float alpha, float beta) {
    for (int i = 0; i < M.rows; i++)
        for (int j = 0; j < M.cols; j++) {
            uchar pixel = M.at<uchar>(i, j);
            int value = alpha * pixel + beta;
            if (value > 255)
                value = 255;
            if (value < 0)
                value = 0;

            uchar &out_pixel = output_M.at<uchar>(i, j);
            out_pixel = (uchar) value;
        }
}


void conv(Mat M, Mat output_M, Mat kernel_v, Mat kernel_h) {
    for (int i = 0; i < M.rows; i++)
        for (int j = 0; j < M.cols; j++) {
            float sum_v = 0, sum_h = 0;
            for (int ki = 0; ki < kernel_v.rows; ki++)
                for (int kj = 0; kj < kernel_v.cols; kj++) {
                    int new_i = i + ki - kernel_v.rows / 2;
                    int new_j = j + kj - kernel_v.cols / 2;

                    if (new_i < 0 || new_i >= M.rows || new_j < 0 || new_j >= M.cols)
                        continue;

                    uchar pixel = M.at<uchar>(new_i, new_j);
                    float kernel_v_pixel = kernel_v.at<float>(ki, kj);
                    float kernel_h_pixel = kernel_h.at<float>(ki, kj);

                    sum_v += kernel_v_pixel * (float) pixel;
                    sum_h += kernel_h_pixel * (float) pixel;
                }


            uchar &out_pixel = output_M.at<uchar>(i, j);
            int value = (int) sqrt(sum_v * sum_v + sum_h * sum_h);
            if (value > 255)
                value = 255;
            if (value < 0)
                value = 0;

            out_pixel = (uchar) value;
        }
}

int main(int argc, char *argv[]) {
    const char *in_path = (const char *) argv[1];
    const char *alpha_arg = (const char *) argv[2];
    const char *beta_arg = (const char *) argv[3];

    float alpha = atof(alpha_arg);
    float beta = atof(beta_arg);

    string adjust_path = "sobel-out-adjust.jpg";
    string out_path = "sobel-out.jpg";

    cout << "Reading image..." << endl;
    Mat M = imread(in_path, IMREAD_GRAYSCALE);

    cout << "Image size: " << M.size() << "  channels: " << M.channels() << " type:" << M.type() << endl;
    Mat adjusted_M = Mat(M.size(), M.type(), Scalar(0));
    Mat output_M = Mat(M.size(), M.type(), Scalar(0));

    cout << "Adjusting brightness and contrast..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    adjust(M, adjusted_M, alpha, beta);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Time of adjustment: " << duration.count() << " ms" << endl;
    

    cout << "Writing adjusted image..." << endl;
    imwrite(adjust_path, adjusted_M);

    float kernel_v_data[] = {-1, 0, 1,
                             -2, 0, 2,
                             -1, 0, 1};

    Mat kernel_v(3, 3, CV_32F, kernel_v_data);

    float kernel_h_data[] = {1, 2, 1,
                             0, 0, 0,
                             -1, -2, -1};

    Mat kernel_h(3, 3, CV_32F, kernel_h_data);

    cout << "Applying filter 1..." << endl;

    start = std::chrono::high_resolution_clock::now();
    conv(adjusted_M, output_M, kernel_v, kernel_h);
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Time of Convolution: " << duration.count() << " ms" << endl;

    cout << "Writing image..." << endl;
    imwrite(out_path, output_M);

    cout << "DONE" << endl;
    return 0;
}