# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -funroll-loops -march=native -std=c++11

# Automatically find all source files
SOURCES = $(wildcard *.cc)

# Automatically create target names by replacing .cc with nothing
TARGETS = $(patsubst %.cc,matmul_%,$(SOURCES))

# Build rules
all: $(TARGETS)

# Pattern rule for creating binaries with matmul_ prefix
matmul_%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean rule
clean:
	rm -f $(TARGETS)

# Phony targets
.PHONY: all clean

