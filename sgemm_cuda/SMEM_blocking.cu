#include <stdio.h>


#ifndef CUDACC
#define CUDACC
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
// Write CUDA kernel for naiive matrix multiplication
template <const uint BLOCKSIZE>
__global__ void sgemm_SMEMblocking(int M, int N, int K, float alpha, const float *A, 
                            const float *B, float beta, float *C) {

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const uint threadRow = threadIdx.x / BLOCKSIZE;
    const uint threadCol = threadIdx.x % BLOCKSIZE;


    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;
    float tmp = 0.0;

    for (int bkIdx = 0; bkIdx < K ; bkIdx += BLOCKSIZE){
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];


        __syncthreads();


        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx){
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * 
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];

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
  sgemm_SMEMblocking<32><<<gridDim, blockDim>>>(M, N, K, 1.0f, dA, dB, 1.0f, dC);
  
  // Record end time and synchronize
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // Copy the result back to Host
  cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

  // Calculate elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed time: %f ms\n", milliseconds);

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  maxRegsPerBlock: %d\n", prop.regsPerBlock);
    printf("  maxRegsPerMultiprocessor: %d\n", prop.regsPerMultiprocessor);
    printf("  MultiProcessor Count: %d\n", prop.multiProcessorCount);

    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  total const memory(Kbytes) %.1f\n",(float)(prop.totalConstMem)/1024.0);
    printf("  memoryBusWidth-Compute Capability: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");

  }
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

