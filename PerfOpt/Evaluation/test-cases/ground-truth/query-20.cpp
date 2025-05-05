#include <iostream>
#include <omp.h>
#include <tuple>
#include <cassert>

using namespace std;

std::tuple<int, int, int, int>  compare_cases(int a, int b, int c, int t);

double G = 2.1;

int main() {
    int original_a = 1;
    int original_b = 2;
    int original_c = 3;
    int original_t = 4;

    auto [a, b, c, t] = compare_cases(original_a, original_b, original_c, original_t);

    std::cout << "Returned values:\n";
    std::cout << "a: " << a << ", b: " << b
              << ", c: " << c << ", t: " << t << "\n";

    assert(a == original_a);
    assert(b == original_b);
    assert(t == original_t);
    assert(c != original_c);

    std::cout << "test case passed!" << endl;

    return 0;
}

std::tuple<int, int, int, int>  compare_cases(int a, int b, int c, int t) {
    omp_set_num_threads(3);
    #pragma omp parallel private(a), firstprivate(b)
    {
        // a will be private and, but not be initialized
        // b will be private and initialized
        // c will be shared;
        // t will be local; that is it is private.
        int t = 5;

        static int s = 8; // will be shared.
        if (omp_get_thread_num() == 0)
            s = 2;

        printf("thread id: %d, a: %d, b: %d, c: %d, t: %d, s: %d, G: %f, \n", omp_get_thread_num(), a, b, c, t, s, G);
        a = 21;
        b = 22;
        c = 23;
        t = 24;
    }

    printf("\nout of the parallel region\n");
    printf("thread id: %d, a: %d, b: %d, c: %d, t: %d, G: %f, \n", omp_get_thread_num(), a, b, c, t, G);

    return std::make_tuple(a, b, c, t);
}