#include <cstdlib> // for std::malloc and std::free
#include <ctime>   // for std::clock
#include <iostream>

#define SIZE 2048
// The kernel function
template <int rows, int columns, int inners>
inline void matmulImplNaive(const float *left, const float *right, float *result)
{
    // Initialize the result matrix to zero
    for (int i = 0; i < rows * columns; ++i)
    {
        result[i] = 0.0f;
    }

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            for (int inner = 0; inner < inners; inner++)
            {
                result[row * columns + col] +=
                    left[row * inners + inner] * right[inner * columns + col];
            }
        }
    }
}

int main()
{
    const int rows = SIZE;    // Set the number of rows
    const int columns = SIZE; // Set the number of columns
    const int inners = SIZE;  // Set the inner dimension

    // Allocate memory for matrices
    float *left = (float *)std::malloc(rows * inners * sizeof(float));
    float *right = (float *)std::malloc(inners * columns * sizeof(float));
    float *result = (float *)std::malloc(rows * columns * sizeof(float));

    // Initialize the left and right matrices with some values
    for (int i = 0; i < rows * inners; ++i)
    {
        left[i] = static_cast<float>(std::rand() % 10); // Random values [0, 10)
    }

    for (int i = 0; i < inners * columns; ++i)
    {
        right[i] = static_cast<float>(std::rand() % 10); // Random values [0, 10)
    }

    // Timing the matrix multiplication
    std::clock_t start = std::clock(); // Start time

    matmulImplNaive<rows, columns, inners>(left, right, result);

    std::clock_t end = std::clock(); // End time

    // Calculate the duration
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds

    std::cout << "Time elapsed: " << duration << std::endl;

    // Print the result
    /*
      std::cout << "Result matrix:\n";
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
          std::cout << result[i * columns + j] << " ";
        }
        std::cout << "\n";
      }
    */

    // Free allocated memory
    std::free(left);
    std::free(right);
    std::free(result);

    return 0;
}
