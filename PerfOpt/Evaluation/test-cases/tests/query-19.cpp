#include <iostream>
#include <omp.h>
#include <cassert>

int sum_heap_elements();

int main() {

    int expected = 3;
    int actual = sum_heap_elements();

    assert(actual == expected && "Sum is not accurate!");

    std::cout << "Test passed!" << std::endl;

    return 0;
}