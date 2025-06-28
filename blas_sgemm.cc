#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cblas.h>  // Requires OpenBLAS or MKL

#ifndef SIZE
#define SIZE 512
#endif

int main() {
    const int N = SIZE;
    const int M = SIZE;
    const int K = SIZE;

    float* A = static_cast<float*>(aligned_alloc(32, M * K * sizeof(float)));
    float* B = static_cast<float*>(aligned_alloc(32, K * N * sizeof(float)));
    float* C = static_cast<float*>(aligned_alloc(32, M * N * sizeof(float)));

    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed\n";
        return 1;
    }

    // Initialize matrices with random values
    for (int i = 0; i < M * K; ++i)
        A[i] = static_cast<float>(std::rand() % 10);

    for (int i = 0; i < K * N; ++i)
        B[i] = static_cast<float>(std::rand() % 10);

    for (int i = 0; i < M * N; ++i)
        C[i] = 0.0f;

    // Measure performance of SGEMM
    std::clock_t start = std::clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    std::clock_t end = std::clock();

    double elapsed_ms = 1000.0 * (end - start) / CLOCKS_PER_SEC / 10.0;
    double gflops = (2.0 * M * N * K) / (elapsed_ms * 1e6);

    std::cout << "Elapsed Time: " << elapsed_ms << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPs/s\n";

    free(A); free(B); free(C);
    return 0;
}
