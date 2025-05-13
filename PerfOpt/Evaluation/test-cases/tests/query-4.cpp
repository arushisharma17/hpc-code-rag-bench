#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <numeric>
#include <algorithm>

// Declare the function
void prefix_sum(int* a, int index, size_t n, size_t block_size, int* flag);

// Helper: compute expected prefix sum (sequential)
std::vector<int> compute_expected_prefix(const std::vector<int>& input) {
    std::vector<int> output(input.size());
    output[0] = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i - 1] + input[i];
    }
    return output;
}

// Unit test runner
void test_prefix_sum(const std::vector<int>& input, size_t block_size, const std::string& label) {
    size_t n = input.size();
    std::vector<int> a = input;
    std::vector<int> expected = compute_expected_prefix(input);

    int num_blocks = (n + block_size - 1) / block_size;
    std::vector<int> flag(num_blocks, 0);

#pragma omp parallel for num_threads(num_blocks)
    for (int b = 0; b < num_blocks; ++b) {
        size_t index = b * block_size;
        size_t current_block_size = std::min(block_size, n - index);
        prefix_sum(a.data(), index, n, current_block_size, flag.data());
    }

    // Validate
    for (size_t i = 0; i < n; ++i) {
        assert(a[i] == expected[i]);
    }

    std::cout << "✅ " << label << " passed.\n";
}

int main() {
    test_prefix_sum({1, 2, 3, 4}, 2, "Test 1: Small input, block size 2");
    test_prefix_sum({5, 10, 15, 20, 25}, 3, "Test 2: Uneven last block");
    test_prefix_sum({42}, 1, "Test 3: Single element");
    test_prefix_sum({0, 0, 0, 0}, 2, "Test 4: All zeros");
    test_prefix_sum({-1, -2, -3, -4}, 2, "Test 5: All negative values");
    
    std::vector<int> large(1000);
    std::iota(large.begin(), large.end(), 1); // 1 to 1000
    test_prefix_sum(large, 64, "Test 6: Large array, block size 64");

    std::cout << "\n🎉 All prefix_sum correctness tests passed.\n";
    return 0;
}
