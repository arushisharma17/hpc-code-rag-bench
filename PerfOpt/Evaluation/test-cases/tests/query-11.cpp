#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

// Prototype of the function under test
void integral_better_reduction();

// Captures output from std::cout
std::string capture_output_better_reduction() {
    std::ostringstream captured;
    auto* old_buf = std::cout.rdbuf();        // Save original buffer
    std::cout.rdbuf(captured.rdbuf());        // Redirect to ostringstream

    integral_better_reduction();              // Call the function

    std::cout.rdbuf(old_buf);                 // Restore original buffer
    return captured.str();                    // Return captured string
}

// Extracts π value from output
double extract_pi_from_output(const std::string& output) {
    std::istringstream iss(output);
    double pi;
    iss >> pi;
    return pi;
}

// Unit test to validate π accuracy
void test_integral_better_reduction_pi_accuracy() {
    std::string output = capture_output_better_reduction();
    double pi = extract_pi_from_output(output);

    double expected_pi = 3.141592653589793;
    double tolerance = 0.01;

    assert(std::abs(pi - expected_pi) < tolerance && "π value from better reduction is not accurate enough");
    std::cout << "✅ test_integral_better_reduction_pi_accuracy passed: π = " << pi << "\n";
}

int main() {
    test_integral_better_reduction_pi_accuracy();
    std::cout << "\n🎉 All better reduction π estimation tests passed.\n";
    return 0;
}
