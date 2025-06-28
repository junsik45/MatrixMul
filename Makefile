# Compiler
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -mavx -fopenmp -funroll-loops -march=native -std=c++11

# Matrix size, overridable via command line
SIZE ?= 512
CXXFLAGS += -DSIZE=$(SIZE)

# BLAS libraries
BLAS_LIBS = -lopenblas

# Sources
SOURCES = $(wildcard *.cc)

# Outputs: matmul_<name>_size<SIZE>
TARGETS = $(patsubst %.cc,matmul_%_size$(SIZE),$(SOURCES))

# Default build
all: $(TARGETS)

# Explicit rule for BLAS files
matmul_blas_sgemm_size$(SIZE): blas_sgemm.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(BLAS_LIBS)

# Rule for other sources
matmul_%_size$(SIZE): %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f matmul_*_size*

