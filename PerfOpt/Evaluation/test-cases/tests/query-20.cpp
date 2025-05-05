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
