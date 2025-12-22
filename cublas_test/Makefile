# Makefile for GEMM Performance Tester

# CUDA toolkit path (adjust if necessary)
CUDA_PATH ?= /usr/local/cuda

# Compiler settings
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++

# Compiler flags
NVCCFLAGS = -O3 -arch=sm_80
CXXFLAGS = -O3
LDFLAGS = -lcublas -lcurand -lcupti -lgmp -lcheckpoint -L../GMP/lib -lcuda -lcudart

# Include directories
INCLUDES = -I../GMP/include -I../NVTX/c/include

# Target executable
TARGET = cublas_test
SOURCE = cublas_test.cu

# Default target
all: $(TARGET)

$(TARGET): $(SOURCE) gmp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Build GMP library using CMake
gmp:
	cd ../GMP && mkdir -p build && cd build && cmake .. && make

# Clean target
clean:
	rm -f $(TARGET)

# Run with default parameters (N=4096, k=16)
run: $(TARGET) gmp
	./$(TARGET)

# Run with small matrices for quick testing (N=1024, k=8)
test: $(TARGET) gmp
	./$(TARGET) 1024 8

# Run with large matrices (N=8192, k=32)
run_large: $(TARGET) gmp
	./$(TARGET) 8192 32

# Run with extra large matrices (N=16384, k=64) - requires significant GPU memory
run_xl: $(TARGET) gmp
	./$(TARGET) 16384 64

# Quick performance comparison
quick: $(TARGET) gmp
	@echo "=== Quick Performance Comparison ==="
	@echo "Small matrices (1024x1024, 8 batches):"
	@./$(TARGET) 1024 8 2 3 | tail -n +12 | head -n 6
	@echo ""
	@echo "Medium matrices (2048x2048, 8 batches):"  
	@./$(TARGET) 2048 8 2 3 | tail -n +12 | head -n 6
	@echo ""
	@echo "Large matrices (4096x4096, 16 batches):"
	@./$(TARGET) 4096 16 2 3 | tail -n +12 | head -n 6

# Show available GPU compute capabilities
gpu_info: $(TARGET)
	@./$(TARGET) 1024 4 1 1 | head -n 15

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build the program"
	@echo "  clean      - Remove built files"
	@echo "  run        - Run with default parameters (N=4096, k=16)"
	@echo "  test       - Quick test with smaller matrices (N=1024, k=8)"
	@echo "  run_large  - Run with large matrices (N=8192, k=32)"
	@echo "  run_xl     - Run with extra large matrices (N=16384, k=64)"
	@echo "  quick      - Quick performance comparison across different sizes"
	@echo "  gpu_info   - Show GPU information"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Manual usage: ./cublas_test [N] [k] [warmup_iter]"
	@echo "  N          - Matrix dimension (must be divisible by k)"
	@echo "  k          - Number of batches"
	@echo "  warmup_iter- Number of warmup iterations (default: 5)"

.PHONY: all clean run test run_large run_xl quick gpu_info help