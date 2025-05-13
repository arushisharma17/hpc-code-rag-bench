#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

// Function prototype (assumes defined elsewhere in same build)
void integral_reduction();

// Captures stdout during function execution
std::string capture_output_reduction() {
    std::ostringstream captured;
    auto* old_buf = std::cout.rdbuf();  // Save original buffer
    std::cout.rdbuf(captured.rdbuf());  // Redirect cout

    integral_reduction();               // Call the function

    std::cout.rdbuf(old_buf);           // Restore cout
    return captured.str();              // Return captured output
}

// Parses and extracts π value from output string
double extract_pi_from_reduction_output(const std::string& output) {
    std::istringstream iss(output);
    double pi;
    iss >> pi;
    return pi;
}

// Unit test: Verify that π is within acceptable bounds
void test_integral_reduction_pi_accuracy() {
    std::string output = capture_output_reduction();
    double pi = extract_pi_from_reduction_output(output);

    double expected_pi = 3.141592653589793;
    double tolerance = 0.01;

    assert(std::abs(pi - expected_pi) < tolerance && "π value from reduction is not within acceptable range");
    std::cout << "✅ test_integral_reduction_pi_accuracy passed: π = " << pi << "\n";
}

int main() {
    test_integral_reduction_pi_accuracy();
    std::cout << "\n🎉 All reduction π estimation tests passed.\n";
    return 0;
}
