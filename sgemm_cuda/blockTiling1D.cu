#include <iostream>


#ifdef CUDACC
#define CUDACC
#endif
#include <cuda.h>
#include <cuda_runtime.h>
#define CEIL_DIV(X,Y) ((X)+(Y-1))/(Y)


template <const int BM, const int BN, const int BK, const int TM>
__global__ void MatMul1DblockTiling(int M, int N, int K, float alpha,
                                  const float *A, const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const int threadRow = threadIdx.x / BN;
    const int threadCol = threadIdx.x % BN ;


    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];

    A += cRow * BM * K; //cRow -> Row index, BM*K Size of the row chunk from original data
    B += cCol * BN; //cCol -> Column index, BN Size of the col chunk from original data
    C += cRow * BM * K + cCol * BN; // output index is determined to be A row and B col


    assert(blockDim.x == BM*BK);
    assert(blockDim.x == BK*BN);
    // figure out inner index
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;


    float threadResults[TM] = {0.0};
    for (uint bkIdx = 0; bkIdx < K ; bkIdx += BK) {

        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();


        A += BK;
        B += BK*N;


        for(uint dotIdx = 0; dotIdx < BK ; ++dotIdx) {

            float Btmp = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {

            threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
            }
        }
        __syncthreads();
    }
  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];

  }
}

int main(void) {

  int M=8192, N=8192, K=8192;
  float *A, *B, *C, *dA, *dB, *dC;
  const int BM=64, BN=64, BK=8, TM=8;
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
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  
  dim3 blockDim((BM*BN)/TM);
  // launch the asynchronous execution of the kernel on the device
  // Record start time
  cudaEventRecord(start);

  // The function call returns immediately on the host
  MatMul1DblockTiling<BM,BN,BK,TM><<<gridDim, blockDim>>>(M, N, K, 1.0f, dA, dB, 1.0f, dC);
  
  // Record end time and synchronize
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copy the result back to Host
  cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  // Clean up
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  free(A);
  free(B);
  free(C);
  return 0;
}
