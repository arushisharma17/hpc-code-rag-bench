#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>

// Prototype
double calculate_pi(int argc, char **argv);

// Tolerance threshold
constexpr double MAX_ALLOWED_ERROR_PCT = 1.0;  // 1%

// Wrapper to test calculate_pi
void run_test(const char* input, const char* description) {
    const char* arg0 = "program_name";
    char* argv[] = { const_cast<char*>(arg0), const_cast<char*>(input) };
    int argc = 2;

    std::cout << "Running test: " << description << " (argv[1] = " << input << ")\n";

    double err = calculate_pi(argc, argv);
    std::cout << "Returned relative error: " << err << " %\n";
    assert(err < MAX_ALLOWED_ERROR_PCT);

    std::cout << "✅ Test passed.\n\n";
}

// Main test runner
int main() {
    run_test("1", "Basic run (low sample)");
    run_test("10", "Moderate sample count");
    run_test("100", "Higher sample count");
    run_test("500", "Stress test with high accuracy");

    std::cout << "🎉 All calculate_pi unit tests passed.\n";
    return 0;
}
