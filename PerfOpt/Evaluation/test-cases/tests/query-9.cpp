#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

// Prototype of the function under test
void integral_atomic();

// Redirects std::cout to a string, calls the function, restores cout
std::string capture_output_atomic() {
    std::ostringstream captured;
    auto* old_buf = std::cout.rdbuf();         // Save original buffer
    std::cout.rdbuf(captured.rdbuf());         // Redirect to captured stream

    integral_atomic();                         // Call the function

    std::cout.rdbuf(old_buf);                  // Restore original buffer
    return captured.str();                     // Return captured output
}

// Extracts the π value from the output string
double extract_pi_from_atomic_output(const std::string& output) {
    std::istringstream iss(output);
    double pi;
    iss >> pi;
    return pi;
}

// Unit test: Checks if π computed via atomic addition is accurate
void test_integral_atomic_pi_accuracy() {
    std::string output = capture_output_atomic();
    double pi = extract_pi_from_atomic_output(output);

    double expected_pi = 3.141592653589793;
    double tolerance = 0.01;

    assert(std::abs(pi - expected_pi) < tolerance && "π value from atomic is not within acceptable tolerance");
    std::cout << "✅ test_integral_atomic_pi_accuracy passed: π = " << pi << "\n";
}

int main() {
    test_integral_atomic_pi_accuracy();
    std::cout << "\n🎉 All atomic π estimation tests passed.\n";
    return 0;
}
