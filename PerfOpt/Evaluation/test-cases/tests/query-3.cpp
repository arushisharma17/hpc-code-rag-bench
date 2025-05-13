#include <iostream>
#include <cassert>

char analyzeCell(char *c_m, int N, int i, int j); // Function prototype

void test_analyzeCell() {
    // Create a 5x5 board (to ensure neighbors exist around (2,2))
    int N = 5;
    char board[25] = {
        '.', '.', '.', '.', '.',  // row 0
        '.', '.', '.', '.', '.',  // row 1
        '.', '.', 'X', '.', '.',  // row 2
        '.', '.', '.', '.', '.',  // row 3
        '.', '.', '.', '.', '.'   // row 4
    };

    // Test: Cell (2,2) is 'X' and has 0 neighbors → should die → '.'
    assert(analyzeCell(board, N, 2, 2) == '.');

    // Add 2 live neighbors → should survive
    board[1 * N + 2] = 'X'; // (1,2)
    board[2 * N + 1] = 'X'; // (2,1)
    assert(analyzeCell(board, N, 2, 2) == 'X');

    // Change center cell to '.' with 3 neighbors → should become alive
    board[2 * N + 2] = '.'; // make center cell dead
    board[3 * N + 2] = 'X'; // (3,2)
    assert(analyzeCell(board, N, 2, 2) == 'X');

    // With more than 3 neighbors → should die
    board[3 * N + 1] = 'X'; // (3,1)
    assert(analyzeCell(board, N, 2, 2) == '.');

    std::cout << "All tests passed.\n";
}

int main() {
    test_analyzeCell();
    return 0;
}
