#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << std::endl;                                    \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define CHECK_CUBLAS(call)                                                     \
    {                                                                          \
        cublasStatus_t stat = (call);                                          \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << stat << std::endl;                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Initialize matrix with random float values
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
}

int main() {
    // Use smaller size for testing, adjust up once verified
    int M = 1024, N = 1024, K = 1024;
    float *A = nullptr, *B = nullptr, *C = nullptr;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    float alpha = 1.0f, beta = 0.0f;

    // Allocate host memory
    A = (float *)malloc(M * K * sizeof(float));
    B = (float *)malloc(K * N * sizeof(float));
    C = (float *)malloc(M * N * sizeof(float));
    if (!A || !B || !C) {
        std::cerr << "Host malloc failed\n";
        return EXIT_FAILURE;
    }

    initialize_matrix(A, M, K);
    initialize_matrix(B, K, N);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));

    // Copy input matrices to device
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record time
    CHECK_CUDA(cudaEventRecord(start));

    // Note: Use CUBLAS_OP_T to treat row-major as transposed column-major
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_T,
                             M, N, K,
                             &alpha,
                             dB, N,  // B transposed
                             dA, K,  // A transposed
                             &beta,
                             dC, M)); // Result in C

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "cuBLAS SGEMM elapsed time: " << milliseconds << " ms" << std::endl;

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(A);
    free(B);
    free(C);

    return 0;
}

