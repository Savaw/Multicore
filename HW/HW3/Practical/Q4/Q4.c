#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef N
#define N 8
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
    gettimeofday(&startTime, NULL);

    for (int i = 0; i < N; i++) {
        int *queens = calloc(N, sizeof(int));
        solve(queens, 0, i);
        free(queens);
    }

    gettimeofday(&stopTime, NULL);
    totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) -
                (startTime.tv_sec * 1000000 + startTime.tv_usec);
    printf("Solution Count: %llu\n", solution_count);
    printf("Time: %ld\n", totalTime);
    return 0;
}
