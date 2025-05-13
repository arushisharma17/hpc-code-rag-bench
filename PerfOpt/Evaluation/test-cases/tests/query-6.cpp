#include <iostream>
#include <sstream>
#include <cassert>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

// Declaration of function under test
void omp_check();

// Capture output printed to std::cout (via printf redirected to std::cout)
std::string capture_omp_check_output() {
    std::ostringstream oss;
    std::streambuf* old_buf = std::cout.rdbuf(); // Save original
    std::cout.rdbuf(oss.rdbuf());                // Redirect to ostringstream

    omp_check();                                 // Call function

    std::cout.rdbuf(old_buf);                    // Restore stdout
    return oss.str();                            // Return captured output
}

void test_omp_check_output_contains_expected_lines() {
    std::string output = capture_omp_check_output();

    // Basic structure checks
    assert(output.find("Info") != std::string::npos);
    assert(output.find("Maximum threads") != std::string::npos);
    assert(output.find("Nested Parallelism") != std::string::npos);

    // Configuration
#ifdef _DEBUG
    assert(output.find("Configuration: Debug") != std::string::npos);
#else
    assert(output.find("Configuration: Release") != std::string::npos);
#endif

    // Platform
#if defined(_M_X64)
    assert(output.find("Platform: x64") != std::string::npos);
#elif defined(_M_IX86)
    assert(output.find("Platform: x86") != std::string::npos);
#endif

    // OpenMP
#ifdef _OPENMP
    assert(output.find("OpenMP is on") != std::string::npos);
    assert(output.find("OpenMP version") != std::string::npos);
#else
    assert(output.find("OpenMP is off") != std::string::npos);
#endif

    std::cout << "✅ omp_check output test passed.\n";
}

int main() {
    test_omp_check_output_contains_expected_lines();
    std::cout << "🎉 All omp_check() tests passed successfully.\n";
    return 0;
}
