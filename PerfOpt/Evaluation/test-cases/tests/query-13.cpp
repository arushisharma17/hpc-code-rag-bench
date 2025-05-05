#include <iostream>
#include <omp.h>
#include <cmath>
#include <cassert>

double integral_atomic();

int main() {
    double pi = integral_atomic();

    const double true_pi = M_PI;
    const double epsilon = 1e-6;

    std::cout << "\nComputed pi: " << pi << ", True pi: " << true_pi << std::endl;

    assert(std::fabs(pi - true_pi) < epsilon && "test case not passed");

    std::cout << "Test passed!" << std::endl;
    return 0;
}