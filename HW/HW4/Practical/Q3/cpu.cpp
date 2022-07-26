#include <iostream>
#include <math.h>
#include <string>
#include <chrono>



#include "opencv2/shape.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;


void invert(Mat M)
{
  for (int i = 0; i < M.rows; i++)
  {
    for (int j = 0; j < M.cols; j++)
    {
      Vec3b &pixel = M.at<Vec3b>(i, j);
      for(int k = 0; k < M.channels(); k++) {
          pixel.val[k] = 255 - pixel.val[k]; 
        } 
    }
  }
}

int main(void)
{
  string in_path = "pics/pic1.jpg";
  string out_path = "out.jpg";
  Mat M = imread(in_path);
  auto start = std::chrono::high_resolution_clock::now();
  invert(M);
  auto stop =  std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "Time: " << duration.count() << "ms" << endl;
  imwrite(out_path, M);
  return 0;
}
