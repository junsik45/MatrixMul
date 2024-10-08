#include <cmath>
#include <cstdlib> // for std::malloc and std::free
#include <ctime>   // for std::clock
#include <iostream>

#include <immintrin.h> // For AVX intrinsics

#define SIZE 512
template <int rows, int columns, int inners>
inline void matmulImplAVX(const float *left, const float *right, float *result)
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
            __m256 vecResult = _mm256_setzero_ps(); // Initialize a zero vector

            for (int inner = 0; inner < inners; inner += 8)
            {
                // Load 8 floats from the left and right matrices
                __m256 vecLeft = _mm256_loadu_ps(&left[row * inners + inner]);
                __m256 vecRight = _mm256_loadu_ps(&right[inner * columns + col]);

                // Multiply the vectors and accumulate the result
                vecResult = _mm256_add_ps(vecResult, _mm256_mul_ps(vecLeft, vecRight));
            }

            // Store the result into the output matrix
            float resultArray[8];
            _mm256_storeu_ps(resultArray, vecResult);

            // Sum the accumulated result from the AVX register
            for (int i = 0; i < 8; ++i)
            {
                result[row * columns + col] += resultArray[i];
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

    // Check for allocation success
    if (left == nullptr || right == nullptr || result == nullptr)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

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

    matmulImplAVX<rows, columns, inners>(left, right, result);

    std::clock_t end = std::clock(); // End time

    // Calculate the duration
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds

    std::cout << "Time elapsed: " << duration << std::endl;
    std::cout << "Performance: " << (2. * pow(SIZE, 3) / duration / 1000000.) << " GFlops/s"
              << std::endl;

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
