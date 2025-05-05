#include <iostream>
#include <omp.h>
#include <cassert>

int sum_heap_elements();

int sum_heap_elements() {
    int heap_sum = 0;
    omp_set_num_threads(3);
    #pragma omp parallel
    {
        int stack_sum=0;
        stack_sum++;
        heap_sum++;
        printf("stack sum is %d\n", stack_sum);
        printf("heap sum is %d\n", heap_sum);
    }

    return heap_sum;
}

int main() {

    int expected = 3;
    int actual = sum_heap_elements();

    assert(actual == expected && "Sum is not accurate!");

    std::cout << "Test passed!" << std::endl;

    return 0;
}