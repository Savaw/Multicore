#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {

    struct timeval startTime, stopTime;
    omp_set_num_threads(16);
    int m;
    int n;
    double tol;
    long totalTime;

    m = atoi(argv[1]);
    n = atoi(argv[2]);
    tol = atof(argv[3]);

    double t[m + 2][n + 2], tnew[m + 1][n + 1], diff, diffmax;

    for (int z = 0; z < 11; z++) {
        gettimeofday(&startTime, NULL);

        #pragma omp parallel default(none) shared(t, n, m)
        {
            int i, j;
            // initialise temperature array
            #pragma omp for collapse(2) schedule(static) // collapse is used to merge two for loops
            for (i = 0; i < m + 2; i++)
                for (j = 0; j < n + 2; j++)
                    t[i][j] = 30.0;
            // fix boundary conditions
            #pragma omp for schedule(static)
            for (i = 1; i <= m; i++) {
                t[i][0] = 10.0;
                t[i][n + 1] = 140.0;
            }

            #pragma omp for schedule(static)
            for (j = 1; j <= n; j++) {
                t[0][j] = 20.0;
                t[m + 1][j] = 100.0;
            }


        }
        // main loop
        int iter = 0;
        diffmax = 1000000.0;
        while (diffmax > tol) {
            iter++;
            #pragma omp parallel default(none) shared(t, tnew, m, n, diffmax, diff)
            {
                int i, j;
                // update temperature for next iteration
                #pragma omp for collapse(2) schedule(static)
                for (i = 1; i <= m; i++)
                    for (j = 1; j <= n; j++)
                        tnew[i][j] =
                                (t[i - 1][j] + t[i + 1][j] + t[i][j - 1] + t[i][j + 1]) / 4.0;

                // work out maximum difference between old and new temperatures
                // Critical is used to prevent multiple threads from updating diffmax at the same time
                #pragma omp critical
                {
                    diffmax = 0.0;
                }
                // collapse(2) is used to merge two for loops. reduction is used to calculate max without race condition.
                #pragma omp for schedule(static) collapse(2) reduction(max:diffmax)
                for (i = 1; i <= m; i++) {
                    for (j = 1; j <= n; j++) {
                        diff = fabs(tnew[i][j] - t[i][j]);
                        if (diff > diffmax) {
                            diffmax = diff;
                        }
                        // copy new to old temperatures
                        t[i][j] = tnew[i][j];
                    }
                }
            }
        }
        gettimeofday(&stopTime, NULL);
        totalTime = (stopTime.tv_sec * 1000000 + stopTime.tv_usec) -
                    (startTime.tv_sec * 1000000 + startTime.tv_usec);

        printf("%ld\n", totalTime);
    }
}
