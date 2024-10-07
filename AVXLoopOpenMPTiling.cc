#include <cmath>
#include <cstdlib> // for std::malloc and std::free
#include <ctime>   // for std::clock
#include <iostream>

#include <immintrin.h> // For AVX intrinsics
#define SIZE 512
#define ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE 128
// The kernel function

template <int rows, int columns, int inners, int tileSize = ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE>
inline void matmulImplAVXRowColParallelInnerTiling(const float *left, const float *right,
                                                   float *result)
{
    // Initialize the result matrix to zero
#pragma omp parallel for
    for (int i = 0; i < rows * columns; ++i)
    {
        result[i] = 0.0f;
    }

#pragma omp parallel for shared(result, left, right) default(none) collapse(3) num_threads(4)
    for (int rowTile = 0; rowTile < rows; rowTile += tileSize)
    {
        for (int innerTile = 0; innerTile < inners; innerTile += tileSize)
        {
            for (int columnTile = 0; columnTile < columns; columnTile += tileSize)
            {
                int rowTileEnd = std::min(rows, rowTile + tileSize);
                for (int row = rowTile; row < rowTileEnd; row++)
                {
                    int innerTileEnd = std::min(inners, innerTile + tileSize);
                    for (int inner = innerTile; inner < innerTileEnd; inner++)
                    {
                        // Load a scalar from the left matrix
                        float leftVal = left[row * inners + inner];
                        __m256 vecLeft =
                            _mm256_set1_ps(leftVal); // Broadcast leftVal across the vector

                        int columnTileEnd = std::min(columns, columnTile + tileSize);
                        for (int col = columnTile; col < columnTileEnd; col += 8)
                        {

                            // Load 8 floats from the right matrix
                            __m256 vecRight = _mm256_loadu_ps(&right[inner * columns + col]);

                            // Load the current result vector (8 floats)
                            __m256 vecResult = _mm256_loadu_ps(&result[row * columns + col]);

                            // Multiply leftVal with vecRight and accumulate into vecResult
                            vecResult = _mm256_fmadd_ps(vecLeft, vecRight, vecResult);

                            // Store the result back
                            _mm256_storeu_ps(&result[row * columns + col], vecResult);
                        }
                    }
                }
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

    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);

    std::clock_t end = std::clock(); // End time

    // Calculate the duration
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC; // in milliseconds

    // Print the result
    /*
    std::cout << "Result matrix:\n";
    for (int i = 0; i < rows * columns; ++i)
    {
        std::cout << result[i] << " ";
        if (i % 1024 == 0)
            std::cout << "\n";
    }
    */

    std::cout << "Time elapsed: " << duration << std::endl;
    std::cout << "Performance: " << (2. * pow(SIZE, 3) / duration / 1000000.) << " GFlops/s"
              << std::endl;

    // Free allocated memory
    std::free(left);
    std::free(right);
    std::free(result);

    return 0;
}