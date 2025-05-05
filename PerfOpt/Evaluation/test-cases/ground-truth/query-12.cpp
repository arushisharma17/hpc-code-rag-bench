#include <iostream>
#include <omp.h>
#include <cmath>
#include <cassert>

double integral_roundrobin();

int main() {
    double pi = integral_roundrobin();

    const double true_pi = M_PI;
    const double epsilon = 1e-6;

    std::cout << "\nComputed pi: " << pi << ", True pi: " << true_pi << std::endl;

    assert(std::fabs(pi - true_pi) < epsilon && "test case not passed");

    std::cout << "Test passed!" << std::endl;
    return 0;
}

double integral_roundrobin() {
    int NTHREADS = 48, nthreads;
    long num_steps = 100000000;
    double step = 0;
    double pi = 0.0;
    double sum[NTHREADS];

    step = 1.0 / (double) num_steps;
    double timer_start = omp_get_wtime();
    omp_set_num_threads(NTHREADS);

    #pragma omp parallel
    {
        int i, id, lnthreads;
        double x;

        lnthreads = omp_get_num_threads();
        id = omp_get_thread_num();
        if (id == 0)
            nthreads = lnthreads;

        for (i = id, sum[id]=0; i < num_steps; i+=lnthreads) {
            x = (i+0.5) * step;
            sum[id] += 4.0 / (1.0 + x*x);
        }

    }
    for (int i = 0; i < nthreads; ++i) {
        pi += sum[i] * step;
    }

    double timer_took = omp_get_wtime() - timer_start;
    std::cout << pi << " took " << timer_took;

    return pi;

    // 1 threads  --> 0.57 seconds.
    // 4 threads  --> 1.34 seconds.
    // 24 threads --> 0.59 seconds.
    // 48 threads --> 0.46 seconds.
}