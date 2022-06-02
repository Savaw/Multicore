#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <sys/time.h>

#ifndef N
#define N 8
#endif

#ifndef NUMBER_OF_THREADS
#define NUMBER_OF_THREADS 16
#endif

unsigned long long int solution_count = 0;

int is_safe(int *queens, int row, int col) {
    for (int i = 0; i < row; i++) {
        if (queens[i] == col) {
            return 0;
        }
        if (abs(queens[i] - col) == row - i) {
            return 0;
        }
    }
    return 1;
}

void solve(int *queens, int row, int col) {
    if (!is_safe(queens, row, col)) {
        return;
    }
    queens[row] = col;
    if (row == N - 1) {
        #pragma omp atomic
        solution_count++;
        return;
    }
    for (int i = 0; i < N; i++) {
        solve(queens, row + 1, i);
    }
}

int main() {
    struct timeval startTime, stopTime;
    long totalTime;
    omp_set_num_threads(NUMBER_OF_THREADS);
    gettimeofday(&startTime, NULL);
    #pragma omp parallel // Parallel is used to enable the usage of task.
    {
        #pragma omp single // the for will be run with only one thread. It will use task to schedule new iterations.

        {
            for (int i = 0; i < N; i++) {
                int *queens = calloc(N, sizeof(int));

                #pragma omp task
                solve(queens, 0, i);
            }
        }
    }
    gettimeofday(&stopTime, NULL);
    totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) -
                (startTime.tv_sec * 1000000 + startTime.tv_usec);
    printf("Solution Count: %llu\n", solution_count);
    printf("Time: %ld\n", totalTime);
    return 0;
}
