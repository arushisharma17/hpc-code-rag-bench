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
