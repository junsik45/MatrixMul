#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Function to initialize a matrix with random values
void initialize_matrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f; // Random values between 0 and 10
    }
}

int main(void) {

  int M=8192, N=8192, K=8192;
  float *A, *B, *C, *dA, *dB, *dC;
  float alpha= 1.0f, beta= 0.0f;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cublasHandle_t handle;
  cublasCreate(&handle);


  
  A = (float *)malloc(M*K*sizeof(float));
  B = (float *)malloc(K*N*sizeof(float));
  C = (float *)malloc(M*N*sizeof(float));


  cudaMalloc(&dA, M*K*sizeof(float));
  cudaMalloc(&dB, K*N*sizeof(float));
  cudaMalloc(&dC, M*N*sizeof(float));

  initialize_matrix(A, M, K);
  initialize_matrix(B, K, N);


  cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice);

  // launch the asynchronous execution of the kernel on the device
  // Record start time
  cudaEventRecord(start);

  // The function call returns immediately on the host
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                   &alpha, dA, M, dB, K, &beta, dC, M);
  
  // Record end time and synchronize
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copy the result back to Host
  cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time:" << milliseconds << " ms" << std::endl;


  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cublasDestroy(handle);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(A);
  free(B);
  free(C);
}

