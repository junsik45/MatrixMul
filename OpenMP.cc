#include <cmath>
#include <cstdlib> // for std::malloc and std::free
#include <ctime>   // for std::clock
#include <iostream>
#define SIZE 512
// The kernel function

template <int rows, int columns, int inners>
inline void matmulImplRowColParallelInnerTiling(const float *left, const float *right,
                                                float *result)
{
#pragma omp parallel for shared(result, left, right) default(none) collapse(2) num_threads(4)
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < columns; col++)
        {
            float sum = 0.0f;
            for (int inner = 0; inner < inners; inner++)
            {
                sum += left[row * inners + inner] * right[inner * columns + col];
            }
            result[row * columns + col] = sum;
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
        left[i] = static_cast<float>(10.0 * std::rand() / RAND_MAX); // Random values [0, 10)
    }

    for (int i = 0; i < inners * columns; ++i)
    {
        right[i] = static_cast<float>(10.0 * std::rand() / RAND_MAX); // Random values [0, 10)
    }

    // Timing the matrix multiplication
    std::clock_t start = std::clock(); // Start time

    matmulImplRowColParallelInnerTiling<rows, columns, inners>(left, right, result);

    std::clock_t end = std::clock(); // End time

    // Calculate the duration
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds

    std::cout << "Time elapsed: " << duration << std::endl;
    std::cout << "Performance: " << (2. * pow(SIZE, 3) / duration / 1000000.) << " GFlops/s"
              << std::endl;

    // Print the result
    std::cout << "Result matrix:\n";
    /*
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j)
        {
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
