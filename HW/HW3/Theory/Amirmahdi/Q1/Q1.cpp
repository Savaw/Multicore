#include <iostream>

// Kernel Dimensions
#define KN 3
#define KM 3

// Input Dimensions
#define IN 3
#define IM 3

float kernel_2d[KN][KM] = {{-1, -2, -1},
                           {0,  0,  0},
                           {1,  2,  1}};
float input[IN][IM] = {{1, 2, 3},
                       {4, 5, 6},
                       {7, 8, 9}};


float flipped_kernel_2d[KN][KM];
float kernel_1d[KN * KM] = {0};
float output[IN][IM] = {0};


template<typename T, size_t N, size_t M>
void printArray(T(&theArray)[N][M]) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < M; y++) {
            std::cout << theArray[x][y] << " ";
        }
        std::cout << std::endl;
    }
}

void flip(float kernel[KN][KM], float flipped_kernel[KN][KM]) {
    for (int i = 0; i < KN; i++) {
        for (int j = 0; j < KM; j++) {
            flipped_kernel[i][j] = kernel[(KN - 1 - i)][(KM - 1 - j)];
        }
    }
}

void flatten_kernel_2d(float kernel[KM][KN], float kernel_1d[KM * KN]) {
    for (int i = 0; i < KN; i++) {
        for (int j = 0; j < KM; j++) {
            kernel_1d[i * KM + j] = kernel[i][j];
        }
    }
}

void map(const float *first, const float *second, float *output, int dim) {
    for (int i = 0; i < dim; i++) {
        output[i] = first[i] * second[i];
    }
}

float map_reduce(float *first, float *second, int dim) {
    float third[dim];
    map(first, second, third, dim);
    float temp = 0;
    for (int i = 0; i < dim; i++) {
        temp += third[i];
    }
    return temp;
}

void convolve_2d(float input[IN][IM], float kernel[KN][KM], float output[IN][IM]) {

    const int dx = KN / 2;
    const int dy = KM / 2;
    float intermediate[KN * KM] = {0};
    flip(kernel, flipped_kernel_2d);
    flatten_kernel_2d(flipped_kernel_2d, kernel_1d);
    for (int i = 0; i < IN; i++) {
        for (int j = 0; j < IM; j++) {
            for (int k = 0; k < KN; k++) {
                for (int l = 0; l < KM; l++) {
                    int x = j - dx + l;
                    int y = i - dy + k;
                    // Branchless conditional
                    intermediate[k * KM + l] = (1 & x >= 0 & x < IM & y >= 0 & y < IN) * input[y][x];
                }
            }
            output[i][j] = map_reduce(intermediate, kernel_1d, KN * KM);
        }
    }
}


int main() {
    convolve_2d(input, kernel_2d, output);
    printArray<float, IN, IM>(output);
    return 0;
}
