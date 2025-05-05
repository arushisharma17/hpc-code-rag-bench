#include <iostream>
#include <omp.h>
#include <cmath>
#include <cassert>
#include <iomanip>

double schedule_dynamic();

int main() {
    const double epsilon = 0.8;
    const double expected = 5e+08;

    const double actual = schedule_dynamic();

    std::cout << "\nComputed avg: " << actual << ", Expected avg: " << expected << std::endl;

    assert(std::fabs(actual - expected) < epsilon && "Average is not accurate enough!");
    std::cout << "Test passed!" << std::endl;
}
