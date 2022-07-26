#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

using namespace std;

#ifndef THREADS_PER_GRID_BLOCK
#define THREADS_PER_GRID_BLOCK 16
#endif

// create max macro
#define MAX(a, b) ((a) > (b) ? (a) : (b))

__global__ void initializerKernel(double *t, int m, int n) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m + 2 && j < n + 2) {
        t[i * (n + 2) + j] = 30.0;
    }

}

__global__ void fixBoundaryConditionKernel1(double *t, int m, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < m + 1) {
        t[i * (n + 2) + 0] = 10.0;
        t[i * (n + 2) + n + 1] = 140.0;
    }

}

__global__ void fixBoundaryConditionKernel2(double *t, int m, int n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > 0 && j < n + 1) {
        t[0 * (n + 2) + j] = 20.0;
        t[(m + 1) * (n + 2) + j] = 100.0;
    }

}

__global__ void main_kernel(const double *t, double *t_new, int m, int n) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < m + 1 && j > 0 && j < n + 1) {
        t_new[i * (n + 1) + j] = (t[(i - 1) * (n + 2) + j] + t[(i + 1) * (n + 2) + j] + t[i * (n + 2) + j - 1] +
                                  t[i * (n + 2) + j + 1]) / 4.0;
    }

}

// Used https://stackoverflow.com/questions/57573872/cuda-reduce-max-min-function-on-matrix-implementation for this reduction
__global__
void
calculate_max_array_and_update_temperature(double *d1, double *d2, double *d_out, unsigned int m_1, unsigned int n_1) {
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c_t = threadIdx.x;
    unsigned int r_t = threadIdx.y;


    unsigned int pos_1D1 = row * (n_1) + col;
    unsigned int pos_1D2 = row * (n_1 + 1) + col;
    unsigned int pos_1D_t = r_t * blockDim.x + c_t;

    extern __shared__ double shared_memory[];


    shared_memory[pos_1D_t] = (row * n_1 + col >= (n_1 + 1) * (m_1 + 1) || row <= 0 || col <= 0 || row >= m_1 ||
                               col >= n_1)
                              ? -9999999.0 // Could be any other negative value.
                              : fabs(
                    d1[pos_1D1] - d2[pos_1D2]);
    if (row < m_1 && col < n_1 && row > 0 && col > 0) {
        d2[pos_1D2] = d1[pos_1D1];
    }
    __syncthreads();


    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {

        if (pos_1D_t < s)
            shared_memory[pos_1D_t] = fmax(shared_memory[pos_1D_t], shared_memory[pos_1D_t + s]);
        __syncthreads();
    }

    if (r_t == 0 && c_t == 0)
        d_out[blockIdx.y * gridDim.x + blockIdx.x] = shared_memory[0];

}



void max_and_update_temperature(double *d1, double *d2, unsigned int m_1, unsigned int n_1, double &result) {

    int threadsPerBlock = THREADS_PER_GRID_BLOCK;
    dim3 blockSize(threadsPerBlock, threadsPerBlock);
    dim3 gridSize((blockSize.x + n_1 - 1) / blockSize.x,
                  (blockSize.y + m_1 - 1) / blockSize.y);

    double *d_out, *d_int;
    cudaMalloc(&d_out, sizeof(double));
    cudaMalloc(&d_int, sizeof(double) * gridSize.y * gridSize.x);

    calculate_max_array_and_update_temperature<<<
    gridSize, blockSize, threadsPerBlock * threadsPerBlock * sizeof(double)>>>(
            d1, d2, d_int, m_1, n_1
    );
    cudaDeviceSynchronize();

    auto *d_int_cpu = static_cast<double *>(malloc(sizeof(double) * gridSize.y * gridSize.x));
    cudaMemcpy(d_int_cpu, d_int, sizeof(double) * gridSize.y * gridSize.x, cudaMemcpyDeviceToHost);
    result = d_int_cpu[0];
    for (int i = 0; i < gridSize.y * gridSize.x; i++) {
        result = MAX(result, d_int_cpu[i]);
    }
    free(d_int_cpu);

    cudaFree(d_out);
    cudaFree(d_int);

}

int main(int argc, char *argv[]) {

    struct timeval startTime{}, stopTime{};

    int m;
    int n;
    double tol; // = 0.0001;
    long totalTime;

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    tol = atof(argv[3]);

    double t[m + 2][n + 2], tnew[m + 1][n + 1], diff, diffmax;
    double *t_gpu, *tnew_gpu;

    cudaMalloc(&t_gpu, sizeof(double) * (m + 2) * (n + 2));
    cudaMalloc(&tnew_gpu, sizeof(double) * (m + 1) * (n + 1));

    for (int z = 0; z < 11; z++) {

        gettimeofday(&startTime, NULL);

        // initialise temperature array
        int threadsPerBlock = THREADS_PER_GRID_BLOCK;
        dim3 gridBlockSize(threadsPerBlock, threadsPerBlock);
        dim3 gridSize((m + 2 + gridBlockSize.x - 1) / gridBlockSize.x,
                      (n + 2 + gridBlockSize.y - 1) / gridBlockSize.y);
        initializerKernel<<<gridSize, gridBlockSize>>>(t_gpu, m, n);
        threadsPerBlock = THREADS_PER_GRID_BLOCK * THREADS_PER_GRID_BLOCK;
        int blocksCount = (m + 2 + threadsPerBlock - 1) / threadsPerBlock;
        fixBoundaryConditionKernel1<<<blocksCount, threadsPerBlock>>>(t_gpu, m, n);
        blocksCount = (n + 2 + threadsPerBlock - 1) / threadsPerBlock;
        fixBoundaryConditionKernel2<<<blocksCount, threadsPerBlock>>>(t_gpu, m, n);
        cudaDeviceSynchronize();

        // main loop
        int iter = 0;
        diffmax = 1000000.0;
        while (diffmax > tol) {
            iter++;
            unsigned int threadsPerBlock = THREADS_PER_GRID_BLOCK;
            dim3 gridBlockSize(threadsPerBlock, threadsPerBlock);
            dim3 gridSize((m + 1 + gridBlockSize.x - 1) / gridBlockSize.x,
                          (n + 1 + gridBlockSize.y - 1) / gridBlockSize.y);
            main_kernel<<<gridSize, gridBlockSize>>>(t_gpu, tnew_gpu, m, n);


            cudaDeviceSynchronize();


            diffmax = 0.0;
            max_and_update_temperature(tnew_gpu, t_gpu, m + 1, n + 1, diffmax);

            cudaDeviceSynchronize();
        }

        cudaMemcpy(t, t_gpu, sizeof(double) * (m + 2) * (n + 2), cudaMemcpyDeviceToHost);
        cudaMemcpy(tnew, tnew_gpu, sizeof(double) * (m + 1) * (n + 1), cudaMemcpyDeviceToHost);

        gettimeofday(&stopTime, NULL);
        totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) -
                    (startTime.tv_sec * 1000000 + startTime.tv_usec);

        printf("%ld\n", totalTime);
    }
    cudaFree(t_gpu);
    cudaFree(tnew_gpu);
}
