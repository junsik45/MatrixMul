#!/bin/bash

source ../env.sh

nvcc -o naive naiive.cu
nvcc -o GMEMcoalescing GMEM_coalescing.cu
nvcc -o SMEMblocking SMEM_blocking.cu
