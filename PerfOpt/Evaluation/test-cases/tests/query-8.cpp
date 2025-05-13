#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

// Function prototype (must match actual definition)
void integral_roundrobin();

// Redirects std::cout to a string, runs the function, and restores cout
std::string capture_output() {
    std::ostringstream captured;
    auto* old_buf = std::cout.rdbuf();         // Save original buffer
    std::cout.rdbuf(captured.rdbuf());         // Redirect to captured stream

    integral_roundrobin();                     // Call function under test

    std::cout.rdbuf(old_buf);                  // Restore original buffer
    return captured.str();                     // Return captured output
}

// Extracts the first number (π) from the string
double extract_pi(const std::string& output) {
    std::istringstream iss(output);
    double pi;
    iss >> pi;
    return pi;
}

// Unit test: Checks if π is accurate
void test_pi_accuracy() {
    std::string output = capture_output();
    double pi = extract_pi(output);

    double expected_pi = 3.141592653589793;
    double tolerance = 0.01;

    assert(std::abs(pi - expected_pi) < tolerance && "π value is not within acceptable tolerance");
    std::cout << "✅ test_pi_accuracy passed: π = " << pi << "\n";
}

int main() {
    test_pi_accuracy();
    std::cout << "\n🎉 All tests completed successfully.\n";
    return 0;
}
