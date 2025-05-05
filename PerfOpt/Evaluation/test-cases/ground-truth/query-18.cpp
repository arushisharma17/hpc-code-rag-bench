#include <iostream>
#include <omp.h>
#include <cmath>
#include <cassert>
#include <iomanip>

double schedule_auto();

int main() {
    const double epsilon = 0.8;
    const double expected = 5e+08;

    const double actual = schedule_auto();

    std::cout << "\nComputed avg: " << actual << ", Expected avg: " << expected << std::endl;

    assert(std::fabs(actual - expected) < epsilon && "Average is not accurate enough!");
    std::cout << "Test passed!" << std::endl;
}

double schedule_auto() {
    int N = 1000000000;
    int i;
    double avg = 0;

    omp_set_num_threads(16);

    double timer_started = omp_get_wtime();

#pragma omp parallel for reduction(+:avg) schedule(auto)
    for (i = 0; i < N; ++i) {
        avg += i;
    }

    avg /= N;

    double elapsed = omp_get_wtime() - timer_started;

    std::cout << avg << " took " << elapsed << std::endl;
    return avg;
}