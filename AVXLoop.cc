#include <cmath>
#include <cstdlib> // for std::malloc and std::free
#include <ctime>   // for std::clock
#include <iostream>

#include <immintrin.h> // For AVX intrinsics

#ifndef SIZE
#define SIZE 512
#endif
template <int rows, int columns, int inners>
inline void matmulOptimizedAVX(const float *left, const float *right, float *result)
{
    // Zero out result matrix using aligned store
    for (int i = 0; i < rows * columns; i += 8)
    {
        _mm256_store_ps(&result[i], _mm256_setzero_ps());
    }

    for (int row = 0; row < rows; row++)
    {
        for (int inner = 0; inner < inners; inner++)
        {
            // Load a scalar from the left matrix
            float leftVal = left[row * inners + inner];
            __m256 vecLeft = _mm256_set1_ps(leftVal); // Broadcast leftVal across the vector

            for (int col = 0; col < columns; col += 8)
            {
                // Load 8 floats from the right matrix
                __m256 vecRight = _mm256_load_ps(&right[inner * columns + col]);

                // Load the current result vector (8 floats)
                __m256 vecResult = _mm256_load_ps(&result[row * columns + col]);

                // Multiply leftVal with vecRight and accumulate into vecResult
                vecResult = _mm256_fmadd_ps(vecLeft, vecRight, vecResult);

                // Store the result back
                _mm256_store_ps(&result[row * columns + col], vecResult);
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
    //float *left = (float *)std::malloc(rows * inners * sizeof(float));
    //float *right = (float *)std::malloc(inners * columns * sizeof(float));
    //float *result = (float *)std::malloc(rows * columns * sizeof(float));
    float* left   = static_cast<float*>(aligned_alloc(32, SIZE * SIZE * sizeof(float)));
    float* right  = static_cast<float*>(aligned_alloc(32, SIZE * SIZE * sizeof(float)));
    float* result = static_cast<float*>(aligned_alloc(32, SIZE * SIZE * sizeof(float)));

    // Check for allocation success
    if (left == nullptr || right == nullptr || result == nullptr)
    {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

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

    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);
    matmulOptimizedAVX<rows, columns, inners>(left, right, result);

    std::clock_t end = std::clock(); // End time

    // Calculate the duration
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC /10.0; // in milliseconds

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
