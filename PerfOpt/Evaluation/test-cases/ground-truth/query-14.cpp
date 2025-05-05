#include <iostream>
#include <omp.h>
#include <cmath>
#include <cassert>

double integral_reduction();

int main() {
    double pi = integral_reduction();

    const double expected = 4;
    const double epsilon = 1e-6;

    std::cout << "\nComputed pi: " << pi << ", expected: " << expected << std::endl;

    assert(std::fabs(pi - expected) < epsilon && "test case not passed");

    std::cout << "Test passed!" << std::endl;
    return 0;
}

double integral_reduction() {
    int NTHREADS = 48;
    long num_steps = 100000000;
    double step = 0;
    double pi = 0.0;
    double sum = 0;
    int i = 0;

    step = 1.0 / (double) num_steps;

    omp_set_num_threads(NTHREADS);
    double timer_start = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (i = 0; i < num_steps; ++i) {
        int x = (i+0.5) * step;
        sum += 4.0 / (1.0 + x*x);
    }

    pi = sum * step;

    double timer_took = omp_get_wtime() - timer_start;
    std::cout << pi << " took " << timer_took;

    return pi;
    // 1 threads  --> 0.55 seconds.
    // 4 threads  --> 0.24 seconds.
    // 24 threads --> 0.24 seconds.
    // 48 threads --> 0.23 seconds.
}