#!/bin/bash

source ../../env.sh

nvcc --ptxas-options=-v -arch=sm_80 -o naive naiive.cu
nvcc --ptxas-options=-v -arch=sm_80 -o GMEMcoalescing GMEM_coalescing.cu
nvcc --ptxas-options=-v -arch=sm_80 -o SMEMblocking SMEM_blocking.cu
