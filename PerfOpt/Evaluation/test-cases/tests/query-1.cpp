#include <iostream>
#include <cassert>
#include <omp.h>

void matrixMult(int *a, int *b, int *c, int N);

// Helper function to run one test case
void runTest(int* A, int* B, int* expected, int N, const std::string& label) {
    int* C = new int[N * N];
    matrixMult(A, B, C, N);
    for (int i = 0; i < N * N; ++i) {
        assert(C[i] == expected[i] && ("Test failed: " + label).c_str());
    }
    std::cout << " Passed: " << label << std::endl;
    delete[] C;
}

int main() {
    {
        int A[] = {1, 2, 3, 4};
        int B[] = {5, 6, 7, 8};
        int E[] = {19, 22, 43, 50};
        runTest(A, B, E, 2, "Basic 2x2 positive integers");
    }

    {
        int A[] = {0, 0, 0, 0};
        int B[] = {5, 6, 7, 8};
        int E[] = {0, 0, 0, 0};
        runTest(A, B, E, 2, "All zero matrix");
    }

    {
        int A[] = {1, 0, 0, 1};
        int B[] = {9, 8, 7, 6};
        int E[] = {9, 8, 7, 6};
        runTest(A, B, E, 2, "Identity matrix (left multiply)");
    }

    {
        int A[] = {2, 4, 6, 8};
        int B[] = {1, 0, 0, 1};
        int E[] = {2, 4, 6, 8};
        runTest(A, B, E, 2, "Identity matrix (right multiply)");
    }

    {
        int A[] = {-1, -2, -3, -4};
        int B[] = {5, 6, 7, 8};
        int E[] = {-19, -22, -43, -50};
        runTest(A, B, E, 2, "Negative values");
    }

    {
        int A[] = {1, -2, -3, 4};
        int B[] = {-1, 2, 3, -4};
        int E[] = {-7, 10, 15, -22};
        runTest(A, B, E, 2, "Mixed positive and negative values");
    }

    {
        int A[] = {7};
        int B[] = {5};
        int E[] = {35};
        runTest(A, B, E, 1, "1x1 edge case");
    }

    {
        int A[] = {1000, 2000, 3000, 4000};
        int B[] = {1, 1, 1, 1};
        int E[] = {3000, 3000, 7000, 7000};
        runTest(A, B, E, 2, "Large values");
    }

    {
        int A[] = {1,1,1, 1,1,1, 1,1,1};
        int B[] = {1,1,1, 1,1,1, 1,1,1};
        int E[] = {3,3,3, 3,3,3, 3,3,3};
        runTest(A, B, E, 3, "All ones 3x3 matrix");
    }

    {
        int A[] = {1,0,0, 0,2,0, 0,0,3};
        int B[] = {4,0,0, 0,5,0, 0,0,6};
        int E[] = {4,0,0, 0,10,0, 0,0,18};
        runTest(A, B, E, 3, "Diagonal matrices");
    }

    std::cout << "All 10 test cases passed successfully!" << std::endl;
    return 0;
}
