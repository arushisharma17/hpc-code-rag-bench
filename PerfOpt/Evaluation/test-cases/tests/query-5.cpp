#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <omp.h>
#include <algorithm>

#define NUM_THREADS 4

// Declare your function
void prefix_sum_parallel(int* a, size_t n, size_t block_size);

// Optional: provide or include implementation of prefix_sum here if needed

// Helper: compute expected prefix sum
std::vector<int> compute_expected_prefix(const std::vector<int>& input) {
    std::vector<int> output(input.size());
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
    return output;
}

// Generic test runner
void test_prefix_sum_parallel(const std::vector<int>& input, size_t block_size, const std::string& label) {
    size_t n = input.size();
    std::vector<int> data = input;
    std::vector<int> expected = compute_expected_prefix(input);

    prefix_sum_parallel(data.data(), n, block_size);

    for (size_t i = 0; i < n; ++i) {
        assert(data[i] == expected[i]);
    }

    std::cout << "✅ " << label << " passed.\n";
}

int main() {
    test_prefix_sum_parallel({1, 2, 3, 4}, 2, "Test 1: 4 elements, block size 2");
    test_prefix_sum_parallel({10, 20, 30, 40, 50}, 3, "Test 2: 5 elements, block size 3");
    test_prefix_sum_parallel({5}, 1, "Test 3: Single element");
    test_prefix_sum_parallel({0, 0, 0, 0}, 2, "Test 4: All zeros");
    test_prefix_sum_parallel({-1, -2, -3, -4}, 2, "Test 5: All negative values");

    std::vector<int> large(1000);
    std::iota(large.begin(), large.end(), 1); // [1, 2, ..., 1000]
    test_prefix_sum_parallel(large, 64, "Test 6: Large array with block size 64");

    std::cout << "\n🎉 All prefix_sum_parallel tests passed.\n";
    return 0;
}
