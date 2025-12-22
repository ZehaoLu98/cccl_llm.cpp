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

# Target executables
TARGET_TRAINING = llama_training_test
SOURCE_TRAINING = llama_training_test.cu
TARGET_INFERENCE = llama_inference_test
SOURCE_INFERENCE = llama_inference_test.cu

# Default target
all: $(TARGET_TRAINING) $(TARGET_INFERENCE)

$(TARGET_TRAINING): $(SOURCE_TRAINING) gmp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(TARGET_INFERENCE): $(SOURCE_INFERENCE) gmp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

# Build GMP library using CMake
gmp:
	cd ../GMP && mkdir -p build && cd build && cmake .. && make

# Clean target
clean:
	rm -f $(TARGET_TRAINING) $(TARGET_INFERENCE)

# Run with default parameters (N=4096, k=16)
run: $(TARGET_TRAINING) gmp
	./$(TARGET_TRAINING)

# Run inference with default parameters
run_inference: $(TARGET_INFERENCE) gmp
	./$(TARGET_INFERENCE)

# Run with small matrices for quick testing (N=1024, k=8)
test: $(TARGET_TRAINING) gmp
	./$(TARGET_TRAINING) 1024 8

# Test inference with small matrices
test_inference: $(TARGET_INFERENCE) gmp
	./$(TARGET_INFERENCE) 1024 8

# Run with large matrices (N=8192, k=32)
run_large: $(TARGET_TRAINING) gmp
	./$(TARGET_TRAINING) 8192 32

# Run inference with large matrices
run_large_inference: $(TARGET_INFERENCE) gmp
	./$(TARGET_INFERENCE) 8192 32

# Run with extra large matrices (N=16384, k=64) - requires significant GPU memory
run_xl: $(TARGET_TRAINING) gmp
	./$(TARGET_TRAINING) 16384 64

# Run inference with extra large matrices
run_xl_inference: $(TARGET_INFERENCE) gmp
	./$(TARGET_INFERENCE) 16384 64

# Quick performance comparison
quick: $(TARGET_TRAINING) gmp
	@echo "=== Quick Performance Comparison ==="
	@echo "Small matrices (1024x1024, 8 batches):"
	@./$(TARGET_TRAINING) 1024 8 2 3 | tail -n +12 | head -n 6
	@echo ""
	@echo "Medium matrices (2048x2048, 8 batches):"  
	@./$(TARGET_TRAINING) 2048 8 2 3 | tail -n +12 | head -n 6
	@echo ""
	@echo "Large matrices (4096x4096, 16 batches):"
	@./$(TARGET_TRAINING) 4096 16 2 3 | tail -n +12 | head -n 6

# Quick performance comparison for inference
quick_inference: $(TARGET_INFERENCE) gmp
	@echo "=== Quick Inference Performance Comparison ==="
	@echo "Small matrices (1024x1024, 8 batches):"
	@./$(TARGET_INFERENCE) 1024 8 2 3 | tail -n +12 | head -n 6
	@echo ""
	@echo "Medium matrices (2048x2048, 8 batches):"  
	@./$(TARGET_INFERENCE) 2048 8 2 3 | tail -n +12 | head -n 6
	@echo ""
	@echo "Large matrices (4096x4096, 16 batches):"
	@./$(TARGET_INFERENCE) 4096 16 2 3 | tail -n +12 | head -n 6

# Show available GPU compute capabilities
gpu_info: $(TARGET_TRAINING)
	@./$(TARGET_TRAINING) 1024 4 1 1 | head -n 15

# Help target
help:
	@echo "Available targets:"
	@echo "  all                - Build both training and inference programs"
	@echo "  clean              - Remove built files"
	@echo "  run                - Run training with default parameters (N=4096, k=16)"
	@echo "  run_inference      - Run inference with default parameters"
	@echo "  test               - Quick training test with smaller matrices (N=1024, k=8)"
	@echo "  test_inference     - Quick inference test with smaller matrices"
	@echo "  run_large          - Run training with large matrices (N=8192, k=32)"
	@echo "  run_large_inference- Run inference with large matrices"
	@echo "  run_xl             - Run training with extra large matrices (N=16384, k=64)"
	@echo "  run_xl_inference   - Run inference with extra large matrices"
	@echo "  quick              - Quick training performance comparison across different sizes"
	@echo "  quick_inference    - Quick inference performance comparison"
	@echo "  gpu_info           - Show GPU information"
	@echo "  help               - Show this help message"
	@echo ""
	@echo "Manual usage: ./llama_training_test [N] [k] [warmup_iter]"
	@echo "Manual usage: ./llama_inference_test [N] [k] [warmup_iter]"
	@echo "  N          - Matrix dimension (must be divisible by k)"
	@echo "  k          - Number of batches"
	@echo "  warmup_iter- Number of warmup iterations (default: 5)"

.PHONY: all clean run run_inference test test_inference run_large run_large_inference run_xl run_xl_inference quick quick_inference gpu_info help