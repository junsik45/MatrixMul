#include <iostream>


#ifndef CUDACC
#define CUDACC
#endif
#include <cuda.h>
#include <cuda_runtime.h>
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
  std::cout << "Elapsed time: " << milliseconds << " ms" << std::endl;

  int nDevices;
  cudaGetDeviceCount(&nDevices);
  
  std::cout << "Number of devices: " << nDevices << std::endl;
  
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::cout << "Device Number: " << i << "\n";
    std::cout << "  Device name: " << prop.name << "\n";
    std::cout << "  maxThreadsPerBlock: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "  maxThreadsPerMultiProcessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "  maxRegsPerBlock: " << prop.regsPerBlock << "\n";
    std::cout << "  maxRegsPerMultiprocessor: " << prop.regsPerMultiprocessor << "\n";
    std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << "\n";

    std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate / 1024 << "\n";
    std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << "\n";
    std::cout << "  Peak Memory Bandwidth (GB/s): " 
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << "\n";
    std::cout << "  Total global memory (Gbytes): " 
              << static_cast<float>(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0 << "\n";
    std::cout << "  Shared memory per block (Kbytes): " 
              << static_cast<float>(prop.sharedMemPerBlock) / 1024.0 << "\n";
    std::cout << "  Total constant memory (Kbytes): " 
              << static_cast<float>(prop.totalConstMem) / 1024.0 << "\n";
    std::cout << "  Compute Capability: " << prop.major << "-" << prop.minor << "\n";
    std::cout << "  Warp-size: " << prop.warpSize << "\n";
    std::cout << "  Concurrent kernels: " << (prop.concurrentKernels ? "yes" : "no") << "\n";
    std::cout << "  Concurrent computation/communication: " 
              << (prop.deviceOverlap ? "yes" : "no") << "\n\n";
  }
    // Get function attributes for myKernel
    cudaFuncAttributes funcAttr;
    cudaError_t err = cudaFuncGetAttributes(&funcAttr, sgemm_SMEMblocking<32>);
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    std::cout << "Registers per thread: " << funcAttr.numRegs << std::endl;
    std::cout << "Shared memory per block (bytes): " << funcAttr.sharedSizeBytes << std::endl;
    std::cout << "Constant memory per block (bytes): " << funcAttr.constSizeBytes << std::endl;
    std::cout << "Local memory per thread (bytes): " << funcAttr.localSizeBytes << std::endl;
    std::cout << "Max threads per block: " << funcAttr.maxThreadsPerBlock << std::endl;
    std::cout << "Binary version: " << funcAttr.binaryVersion << std::endl;
    std::cout << "Cache mode CA: " << funcAttr.cacheModeCA << std::endl;
    std::cout << "Preferred shared memory carve-out: " << funcAttr.preferredShmemCarveout << std::endl;
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

