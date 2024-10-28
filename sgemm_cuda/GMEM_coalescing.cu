#include <stdio.h>


// Write CUDA kernel for naiive matrix multiplication
template <const uint BLOCKSIZE>
__global__ void sgemm_GMEMcoalescing(int M, int N, int K, float alpha, const float *A, 
                            const float *B, float beta, float *C) {

    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K ; ++i){
        tmp += A[ x * K + i ] * B[i*N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x*N + y];
    }

}


int main(void) {

  int M=8192, N=8192, K=8192;
  float *A, *B, *C, *dA, *dB, *dC;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  
  A = (float *)malloc(M*K*sizeof(float));
  B = (float *)malloc(K*N*sizeof(float));
  C = (float *)malloc(M*N*sizeof(float));


  cudaMalloc(&dA, M*K*sizeof(float));
  cudaMalloc(&dB, K*N*sizeof(float));
  cudaMalloc(&dC, M*N*sizeof(float));


  cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice);

  // create as many blocks as necessary to map all of C
  dim3 gridDim((M + 31)/ 32, (N+31)/ 32, 1);
  
  dim3 blockDim(32, 32, 1);
  // launch the asynchronous execution of the kernel on the device
  // Record start time
  cudaEventRecord(start);

  // The function call returns immediately on the host
  sgemm_GMEMcoalescing<32><<<gridDim, blockDim>>>(M, N, K, 1.0f, dA, dB, 1.0f, dC);
  
  // Record end time and synchronize
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copy the result back to Host
  cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed time: %f ms\n", milliseconds);

  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(A);
  free(B);
  free(C);
}

