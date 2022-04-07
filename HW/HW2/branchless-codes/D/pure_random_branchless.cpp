// TODO

#include "benchmark/benchmark.h"
#include <iostream>
#include <stdlib.h>
#include <vector>

void BM_branch_not_pridicted(benchmark::State &state) {
    srand(1);
    const unsigned int N = state.range(0);
    std::vector<unsigned long> v1(N);
    std::vector<unsigned long> v2(N);
    std::vector<int> c(N);
    for (size_t i = 0; i < N; ++i) {
        v1[i] = rand();
        v2[i] = rand();
        c[i] = rand() & 0x1;
    }
    unsigned long *p1 = v1.data();
    unsigned long *p2 = v2.data();
    int *c_ptr = c.data();
    for (auto _ : state) {
        unsigned long a = 0;
        for (size_t i = 0; i < N; ++i) {
            bool flag= c_ptr[i]!=0;
            a = (a+ p1[i] * (flag)) * (p2[i] * (!flag) + 1 * (flag));
//            if (c_ptr[i]) {
//                a += p1[i];
//            } else {
//                a *= p2[i];
//            }
        }
        benchmark::DoNotOptimize(a);
    }
    benchmark::ClobberMemory();
    state.SetItemsProcessed(N*state.iterations());
}
BENCHMARK(BM_branch_not_pridicted)->Arg(1e6);

