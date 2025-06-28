#include <cmath>
#include <cstdlib> // for std::malloc and std::free
#include <ctime>   // for std::clock
#include <iostream>

#include <immintrin.h> // For AVX intrinsics
#ifndef SIZE
#define SIZE 512
#endif
#define ROW_COL_PARALLEL_INNER_TILING_TILE_SIZE 32
// The kernel function
inline void microkernel_4x8(const float* left, const float* right, float* result,
                            int lda, int ldb, int ldc, int k)
{
    __m256 c0 = _mm256_load_ps(&result[0 * ldc]); // Row 0
    __m256 c1 = _mm256_load_ps(&result[1 * ldc]); // Row 1
    __m256 c2 = _mm256_load_ps(&result[2 * ldc]); // Row 2
    __m256 c3 = _mm256_load_ps(&result[3 * ldc]); // Row 3

    for (int p = 0; p < k; ++p)
    {

        if (p + 4 < k) {
            _mm_prefetch((const char*)&left[0 * lda + p + 4], _MM_HINT_T0);
            _mm_prefetch((const char*)&left[1 * lda + p + 4], _MM_HINT_T0);
            _mm_prefetch((const char*)&left[2 * lda + p + 4], _MM_HINT_T0);
            _mm_prefetch((const char*)&left[3 * lda + p + 4], _MM_HINT_T0);

            _mm_prefetch((const char*)&right[(p + 4) * ldb], _MM_HINT_T0);
        }

        __m256 b = _mm256_load_ps(&right[p * ldb]); // 8 floats from B

        __m256 a0 = _mm256_set1_ps(left[0 * lda + p]);
        __m256 a1 = _mm256_set1_ps(left[1 * lda + p]);
        __m256 a2 = _mm256_set1_ps(left[2 * lda + p]);
        __m256 a3 = _mm256_set1_ps(left[3 * lda + p]);

        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }

    _mm256_store_ps(&result[0 * ldc], c0);
    _mm256_store_ps(&result[1 * ldc], c1);
    _mm256_store_ps(&result[2 * ldc], c2);
    _mm256_store_ps(&result[3 * ldc], c3);
}


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

#pragma omp parallel for shared(result, left, right) default(none) collapse(3) schedule(static)
    for (int rowTile = 0; rowTile < rows; rowTile += tileSize)
    {
        for (int innerTile = 0; innerTile < inners; innerTile += tileSize)
        {
            for (int columnTile = 0; columnTile < columns; columnTile += tileSize)
            {
                int innerTileEnd = std::min(inners, innerTile + tileSize);
                int rowTileEnd = std::min(rows, rowTile + tileSize);
                int columnTileEnd = std::min(columns, columnTile + tileSize);
                for (int row = rowTile; row + 3 < rowTileEnd; row += 4)
                {
                    for (int col = columnTile; col + 7 < columnTileEnd; col += 8)
                    {
                        microkernel_4x8(&left[row * inners + innerTile],
                                    &right[innerTile * columns + col],
                                    &result[row * columns + col],
                                    inners, columns, columns, innerTileEnd - innerTile);
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
    float* left   = static_cast<float*>(aligned_alloc(32, rows * inners * sizeof(float)));
    float* right  = static_cast<float*>(aligned_alloc(32, inners * columns * sizeof(float)));
    float* result = static_cast<float*>(aligned_alloc(32, rows * columns * sizeof(float)));

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
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);
    matmulImplAVXRowColParallelInnerTiling<rows, columns, inners>(left, right, result);

    std::clock_t end = std::clock(); // End time

    // Calculate the duration
    double duration = 1000.0 * (end - start) / CLOCKS_PER_SEC / 10.0; // in milliseconds

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
