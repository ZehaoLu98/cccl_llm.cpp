#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include <gmp/profile.h>
#include <cupti_checkpoint.h>
using NV::Cupti::Checkpoint::CUpti_Checkpoint;

__global__ void helloKernel() {
    // Simple kernel to ensure GPU is active
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        printf("Hello from GPU!\n");
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define TRY_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s - skipping test\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while(0)

#define TRY_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("cuBLAS error at %s:%d: %d - skipping test\n", __FILE__, __LINE__, status); \
            return false; \
        } \
    } while(0)

class GemmPerformanceTester {
private:
    cublasHandle_t handle;
    int N;
    int k;
    int warmup_iterations;
    
    // Memory bandwidth calculation helpers
    size_t getMemoryFootprint(int m, int n, int k) {
        return sizeof(float) * (m * k + k * n + m * n); // A + B + C matrices
    }
    
    double calculateThroughput(size_t bytes, double time_ms) {
        return (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0); // GB/s
    }
    
    double calculateGFLOPS(long long ops, double time_ms) {
        return (ops / 1e9) / (time_ms / 1000.0);
    }
    
    void printHeader() {
        printf("\n%-20s %-15s %-15s %-15s\n", 
               "Test Type", "Time (ms)", "DRAM GB/s", "Compute GFLOPS");
        printf("%-20s %-15s %-15s %-15s\n", 
               "--------------------", "---------------", "---------------", "---------------");
    }
    
public:
    GemmPerformanceTester(int N, int k, int warmup_iter = 5) 
        : N(N), k(k), warmup_iterations(warmup_iter) {
        CHECK_CUBLAS(cublasCreate(&handle));
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    }
    
    ~GemmPerformanceTester() {
        cublasDestroy(handle);
    }
    
    bool testNaiveGemm() {
        printf("\n=== Testing Naive GEMM (%d individual %dx%d multiplications) ===\n", k, N/k, N);
        
        int batch_size = k;
        int m = N;          // Height of A and C
        int n = N;          // Width of B and C  
        int k_dim = 128;    // Width of A and height of B
        
        // Allocate matrices
        std::vector<float*> d_A_array(batch_size), d_B_array(batch_size), d_C_array(batch_size);
        
        size_t size_A = m * k_dim * sizeof(float);
        size_t size_B = k_dim * n * sizeof(float);
        size_t size_C = m * n * sizeof(float);
        
        // Allocate individual matrices
        for (int i = 0; i < batch_size; i++) {
            if (cudaMalloc(&d_A_array[i], size_A) != cudaSuccess ||
                cudaMalloc(&d_B_array[i], size_B) != cudaSuccess ||
                cudaMalloc(&d_C_array[i], size_C) != cudaSuccess) {
                printf("CUDA malloc failed for batch %d - skipping test\n", i);
                // Cleanup already allocated memory
                for (int j = 0; j <= i; j++) {
                    if (d_A_array[j]) cudaFree(d_A_array[j]);
                    if (d_B_array[j]) cudaFree(d_B_array[j]);
                    if (d_C_array[j]) cudaFree(d_C_array[j]);
                }
                return false;
            }
        }
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        
        for (int i = 0; i < batch_size; i++) {
            curandGenerateUniform(gen, d_A_array[i], m * k_dim);
            curandGenerateUniform(gen, d_B_array[i], k_dim * n);
            curandGenerateUniform(gen, d_C_array[i], m * n);
        }
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int iter = 0; iter < warmup_iterations; iter++) {
            for (int i = 0; i < batch_size; i++) {
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                       m, n, k_dim,
                                       &alpha,
                                       d_A_array[i], m,
                                       d_B_array[i], k_dim,
                                       &beta,
                                       d_C_array[i], m));
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("NaiveGEMM", GmpProfileType::CONCURRENT_KERNEL);
        for (int i = 0; i < batch_size; i++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   m, n, k_dim,
                                   &alpha,
                                   d_A_array[i], m,
                                   d_B_array[i], k_dim,
                                   &beta,
                                   d_C_array[i], m));
        }
        GmpProfiler::getInstance()->popRange("NaiveGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t total_memory_bytes = batch_size * (size_A + size_B + size_C);
        long long total_flops = batch_size * 2LL * m * n * k_dim;
        
        double throughput = calculateThroughput(total_memory_bytes, time_ms);
        double gflops = calculateGFLOPS(total_flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Naive GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        for (int i = 0; i < batch_size; i++) {
            CHECK_CUDA(cudaFree(d_A_array[i]));
            CHECK_CUDA(cudaFree(d_B_array[i]));
            CHECK_CUDA(cudaFree(d_C_array[i]));
        }
        
        curandDestroyGenerator(gen);
        return true;
    }

    bool testBatchedGemm() {
        printf("\n=== Testing Batched GEMM (%d batches of %dx%d) ===\n", k, N/k, N);
        
        int batch_size = k;
        int m = N;          // Height of A and C
        int n = N;          // Width of B and C  
        int k_dim = 128;    // Width of A and height of B
        
        // Allocate matrices
        std::vector<float*> h_A(batch_size), h_B(batch_size), h_C(batch_size);
        std::vector<float*> d_A_array(batch_size), d_B_array(batch_size), d_C_array(batch_size);
        
        size_t size_A = m * k_dim * sizeof(float);
        size_t size_B = k_dim * n * sizeof(float);
        size_t size_C = m * n * sizeof(float);
        
        // Allocate individual matrices
        for (int i = 0; i < batch_size; i++) {
            if (cudaMalloc(&d_A_array[i], size_A) != cudaSuccess ||
                cudaMalloc(&d_B_array[i], size_B) != cudaSuccess ||
                cudaMalloc(&d_C_array[i], size_C) != cudaSuccess) {
                printf("CUDA malloc failed for batch %d in testBatchedGemm - skipping test\n", i);
                for (int j = 0; j <= i; j++) {
                    if (d_A_array[j]) cudaFree(d_A_array[j]);
                    if (d_B_array[j]) cudaFree(d_B_array[j]);
                    if (d_C_array[j]) cudaFree(d_C_array[j]);
                }
                return false;
            }
        }
        
        // Copy pointers to device
        float **d_A_ptr, **d_B_ptr, **d_C_ptr;
        if (cudaMalloc(&d_A_ptr, batch_size * sizeof(float*)) != cudaSuccess ||
            cudaMalloc(&d_B_ptr, batch_size * sizeof(float*)) != cudaSuccess ||
            cudaMalloc(&d_C_ptr, batch_size * sizeof(float*)) != cudaSuccess) {
            printf("CUDA malloc failed for pointer arrays in testBatchedGemm - skipping test\n");
            for (int i = 0; i < batch_size; i++) {
                cudaFree(d_A_array[i]);
                cudaFree(d_B_array[i]);
                cudaFree(d_C_array[i]);
            }
            return false;
        }
        
        CHECK_CUDA(cudaMemcpy(d_A_ptr, d_A_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_ptr, d_B_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C_ptr, d_C_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        
        for (int i = 0; i < batch_size; i++) {
            curandGenerateUniform(gen, d_A_array[i], m * k_dim);
            curandGenerateUniform(gen, d_B_array[i], k_dim * n);
            curandGenerateUniform(gen, d_C_array[i], m * n);
        }
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int iter = 0; iter < warmup_iterations; iter++) {
            CHECK_CUBLAS(cublasSgemmBatched(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          m, n, k_dim,
                                          &alpha,
                                          (const float**)d_A_ptr, m,
                                          (const float**)d_B_ptr, k_dim,
                                          &beta,
                                          d_C_ptr, m,
                                          batch_size));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("BatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUBLAS(cublasSgemmBatched(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, n, k_dim,
                                      &alpha,
                                      (const float**)d_A_ptr, m,
                                      (const float**)d_B_ptr, k_dim,
                                      &beta,
                                      d_C_ptr, m,
                                      batch_size));
        GmpProfiler::getInstance()->popRange("BatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t total_memory_bytes = batch_size * (size_A + size_B + size_C);
        long long total_flops = batch_size * 2LL * m * n * k_dim;
        
        double throughput = calculateThroughput(total_memory_bytes, time_ms);
        double gflops = calculateGFLOPS(total_flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Batched GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        for (int i = 0; i < batch_size; i++) {
            CHECK_CUDA(cudaFree(d_A_array[i]));
            CHECK_CUDA(cudaFree(d_B_array[i]));
            CHECK_CUDA(cudaFree(d_C_array[i]));
        }
        
        CHECK_CUDA(cudaFree(d_A_ptr));
        CHECK_CUDA(cudaFree(d_B_ptr));
        CHECK_CUDA(cudaFree(d_C_ptr));
        curandDestroyGenerator(gen);
        return true;
    }
    
    bool testStridedBatchedGemm() {
        printf("\n=== Testing Strided Batched GEMM (%d batches of %dx%d) ===\n", k, N/k, N);
        
        int batch_size = k;
        int m = N;          // Height of A and C
        int n = N;          // Width of B and C
        int k_dim = 128;    // Width of A and height of B
        
        // Calculate strides
        long long stride_A = m * k_dim;
        long long stride_B = k_dim * n;
        long long stride_C = m * n;
        
        // Allocate strided matrices
        float *d_A, *d_B, *d_C;
        size_t total_size_A = stride_A * batch_size * sizeof(float);
        size_t total_size_B = stride_B * batch_size * sizeof(float);
        size_t total_size_C = stride_C * batch_size * sizeof(float);
        
        if (cudaMalloc(&d_A, total_size_A) != cudaSuccess ||
            cudaMalloc(&d_B, total_size_B) != cudaSuccess ||
            cudaMalloc(&d_C, total_size_C) != cudaSuccess) {
            printf("CUDA malloc failed in testStridedBatchedGemm - skipping test\n");
            if (d_A) cudaFree(d_A);
            if (d_B) cudaFree(d_B);
            if (d_C) cudaFree(d_C);
            return false;
        }
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_A, stride_A * batch_size);
        curandGenerateUniform(gen, d_B, stride_B * batch_size);
        curandGenerateUniform(gen, d_C, stride_C * batch_size);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int iter = 0; iter < warmup_iterations; iter++) {
            CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 m, n, k_dim,
                                                 &alpha,
                                                 d_A, m, stride_A,
                                                 d_B, k_dim, stride_B,
                                                 &beta,
                                                 d_C, m, stride_C,
                                                 batch_size));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("StridedBatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                             CUBLAS_OP_N, CUBLAS_OP_N,
                                             m, n, k_dim,
                                             &alpha,
                                             d_A, m, stride_A,
                                             d_B, k_dim, stride_B,
                                             &beta,
                                             d_C, m, stride_C,
                                             batch_size));
        GmpProfiler::getInstance()->popRange("StridedBatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t total_memory_bytes = total_size_A + total_size_B + total_size_C;
        long long total_flops = batch_size * 2LL * m * n * k_dim;
        
        double throughput = calculateThroughput(total_memory_bytes, time_ms);
        double gflops = calculateGFLOPS(total_flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Strided Batch GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        curandDestroyGenerator(gen);
        return true;
    }
    
    bool testBigGemm_v2() {
        printf("\n=== Testing Big GEMM (%dx%d) ===\n", N, N * k);
        
        // Allocate matrices
        float *d_A, *d_B, *d_C;
        size_t size = N * N * k * sizeof(float);
        
        TRY_CUDA(cudaMalloc(&d_A, size));
        TRY_CUDA(cudaMalloc(&d_B, size));
        TRY_CUDA(cudaMalloc(&d_C, size));
        
        // Initialize with random data
        curandGenerator_t gen;
        if (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
            printf("cuRAND error: failed to create generator - skipping test\n");
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return false;
        }
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_A, N * N);
        curandGenerateUniform(gen, d_B, N * N * k);
        curandGenerateUniform(gen, d_C, N * N * k);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int i = 0; i < warmup_iterations; i++) {
            TRY_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   N, N*k, N,
                                   &alpha,
                                   d_A, N,
                                   d_B, N,
                                   &beta,
                                   d_C, N));
        }
        TRY_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        TRY_CUDA(cudaEventCreate(&start));
        TRY_CUDA(cudaEventCreate(&stop));
        
        TRY_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("BigGEMM", GmpProfileType::CONCURRENT_KERNEL);
        TRY_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, N*k, N,
                               &alpha,
                               d_A, N,
                               d_B, N,
                               &beta,
                               d_C, N));
        GmpProfiler::getInstance()->popRange("BigGEMM", GmpProfileType::CONCURRENT_KERNEL);
        TRY_CUDA(cudaEventRecord(stop, 0));
        TRY_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        TRY_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t memory_bytes = 3 * size; // Read A, B; Write C
        long long flops = 2LL * N * N * N; // 2*N^3 operations for matrix multiply
        
        double throughput = calculateThroughput(memory_bytes, time_ms);
        double gflops = calculateGFLOPS(flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Big GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        curandDestroyGenerator(gen);
        return true;
    }
    
    bool testNaiveGemm_v2() {
        printf("\n=== Testing Naive GEMM (%d individual %dx%d multiplications) ===\n", k, N, 128);
        
        int batch_size = k;
        int m = N;          // Height of A and C
        int n = N;          // Width of B and C  
        int k_dim = 128;    // Width of A and height of B
        
        // Allocate matrices
        std::vector<float*> d_A_array(batch_size), d_B_array(batch_size), d_C_array(batch_size);
        
        size_t size_A = m * k_dim * sizeof(float);
        size_t size_B = k_dim * n * sizeof(float);
        size_t size_C = m * n * sizeof(float);
        
        // Allocate individual matrices
        for (int i = 0; i < batch_size; i++) {
            if (cudaMalloc(&d_A_array[i], size_A) != cudaSuccess ||
                cudaMalloc(&d_B_array[i], size_B) != cudaSuccess ||
                cudaMalloc(&d_C_array[i], size_C) != cudaSuccess) {
                printf("CUDA malloc failed for batch %d in testNaiveGemm_v2 - skipping test\n", i);
                for (int j = 0; j <= i; j++) {
                    if (d_A_array[j]) cudaFree(d_A_array[j]);
                    if (d_B_array[j]) cudaFree(d_B_array[j]);
                    if (d_C_array[j]) cudaFree(d_C_array[j]);
                }
                return false;
            }
        }
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        
        for (int i = 0; i < batch_size; i++) {
            curandGenerateUniform(gen, d_A_array[i], m * k_dim);
            curandGenerateUniform(gen, d_B_array[i], k_dim * n);
            curandGenerateUniform(gen, d_C_array[i], m * n);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int iter = 0; iter < warmup_iterations; iter++) {
            for (int i = 0; i < batch_size; i++) {
                CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                       m, n, k_dim,
                                       &alpha,
                                       d_A_array[i], m,
                                       d_B_array[i], k_dim,
                                       &beta,
                                       d_C_array[i], m));
            }
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("NaiveGEMM", GmpProfileType::CONCURRENT_KERNEL);
        for (int i = 0; i < batch_size; i++) {
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   m, n, k_dim,
                                   &alpha,
                                   d_A_array[i], m,
                                   d_B_array[i], k_dim,
                                   &beta,
                                   d_C_array[i], m));
        }
        GmpProfiler::getInstance()->popRange("NaiveGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t total_memory_bytes = batch_size * (size_A + size_B + size_C);
        long long total_flops = batch_size * 2LL * m * n * k_dim;
        
        double throughput = calculateThroughput(total_memory_bytes, time_ms);
        double gflops = calculateGFLOPS(total_flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Naive GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        for (int i = 0; i < batch_size; i++) {
            CHECK_CUDA(cudaFree(d_A_array[i]));
            CHECK_CUDA(cudaFree(d_B_array[i]));
            CHECK_CUDA(cudaFree(d_C_array[i]));
        }
        
        curandDestroyGenerator(gen);
        return true;
    }

    bool testBatchedGemm_v2() {
        printf("\n=== Testing Batched GEMM (%d batches of %dx%d) ===\n", k, N, N);
        
        int batch_size = k;
        int m = N;          // Height of A and C
        int n = N;          // Width of B and C  
        int k_dim = 128;    // Width of A and height of B
        
        // Allocate matrices
        std::vector<float*> h_A(batch_size), h_B(batch_size), h_C(batch_size);
        std::vector<float*> d_A_array(batch_size), d_B_array(batch_size), d_C_array(batch_size);
        
        size_t size_A = m * k_dim * sizeof(float);
        size_t size_B = k_dim * n * sizeof(float);
        size_t size_C = m * n * sizeof(float);
        
        // Allocate individual matrices
        for (int i = 0; i < batch_size; i++) {
            if (cudaMalloc(&d_A_array[i], size_A) != cudaSuccess ||
                cudaMalloc(&d_B_array[i], size_B) != cudaSuccess ||
                cudaMalloc(&d_C_array[i], size_C) != cudaSuccess) {
                printf("CUDA malloc failed for batch %d in testBatchedGemm_v2 - skipping test\n", i);
                for (int j = 0; j <= i; j++) {
                    if (d_A_array[j]) cudaFree(d_A_array[j]);
                    if (d_B_array[j]) cudaFree(d_B_array[j]);
                    if (d_C_array[j]) cudaFree(d_C_array[j]);
                }
                return false;
            }
        }
        
        // Copy pointers to device
        float **d_A_ptr, **d_B_ptr, **d_C_ptr;
        if (cudaMalloc(&d_A_ptr, batch_size * sizeof(float*)) != cudaSuccess ||
            cudaMalloc(&d_B_ptr, batch_size * sizeof(float*)) != cudaSuccess ||
            cudaMalloc(&d_C_ptr, batch_size * sizeof(float*)) != cudaSuccess) {
            printf("CUDA malloc failed for pointer arrays in testBatchedGemm_v2 - skipping test\n");
            for (int i = 0; i < batch_size; i++) {
                cudaFree(d_A_array[i]);
                cudaFree(d_B_array[i]);
                cudaFree(d_C_array[i]);
            }
            return false;
        }
        
        CHECK_CUDA(cudaMemcpy(d_A_ptr, d_A_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B_ptr, d_B_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C_ptr, d_C_array.data(), batch_size * sizeof(float*), cudaMemcpyHostToDevice));
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        
        for (int i = 0; i < batch_size; i++) {
            curandGenerateUniform(gen, d_A_array[i], m * k_dim);
            curandGenerateUniform(gen, d_B_array[i], k_dim * n);
            curandGenerateUniform(gen, d_C_array[i], m * n);
        }
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int iter = 0; iter < warmup_iterations; iter++) {
            CHECK_CUBLAS(cublasSgemmBatched(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          m, n, k_dim,
                                          &alpha,
                                          (const float**)d_A_ptr, m,
                                          (const float**)d_B_ptr, k_dim,
                                          &beta,
                                          d_C_ptr, m,
                                          batch_size));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("BatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUBLAS(cublasSgemmBatched(handle,
                                      CUBLAS_OP_N, CUBLAS_OP_N,
                                      m, n, k_dim,
                                      &alpha,
                                      (const float**)d_A_ptr, m,
                                      (const float**)d_B_ptr, k_dim,
                                      &beta,
                                      d_C_ptr, m,
                                      batch_size));
        GmpProfiler::getInstance()->popRange("BatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t total_memory_bytes = batch_size * (size_A + size_B + size_C);
        long long total_flops = batch_size * 2LL * m * n * k_dim;
        
        double throughput = calculateThroughput(total_memory_bytes, time_ms);
        double gflops = calculateGFLOPS(total_flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Batched GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        
        for (int i = 0; i < batch_size; i++) {
            CHECK_CUDA(cudaFree(d_A_array[i]));
            CHECK_CUDA(cudaFree(d_B_array[i]));
            CHECK_CUDA(cudaFree(d_C_array[i]));
        }
        
        CHECK_CUDA(cudaFree(d_A_ptr));
        CHECK_CUDA(cudaFree(d_B_ptr));
        CHECK_CUDA(cudaFree(d_C_ptr));
        curandDestroyGenerator(gen);
        return true;
    }
    
    bool testStridedBatchedGemm_v2() {
        printf("\n=== Testing Strided Batched GEMM (%d batches of %dx%d) ===\n", k, N, N);
        
        int batch_size = k;
        int m = N;          // Height of A and C
        int n = N;          // Width of B and C
        int k_dim = 128;    // Width of A and height of B
        
        // Calculate strides
        long long stride_A = m * k_dim;
        long long stride_B = k_dim * n;
        long long stride_C = m * n;
        
        // Allocate strided matrices
        float *d_A, *d_B, *d_C;
        size_t total_size_A = stride_A * batch_size * sizeof(float);
        size_t total_size_B = stride_B * batch_size * sizeof(float);
        size_t total_size_C = stride_C * batch_size * sizeof(float);
        
        if (cudaMalloc(&d_A, total_size_A) != cudaSuccess ||
            cudaMalloc(&d_B, total_size_B) != cudaSuccess ||
            cudaMalloc(&d_C, total_size_C) != cudaSuccess) {
            printf("CUDA malloc failed in testStridedBatchedGemm_v2 - skipping test\n");
            if (d_A) cudaFree(d_A);
            if (d_B) cudaFree(d_B);
            if (d_C) cudaFree(d_C);
            return false;
        }
        
        // Initialize with random data
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_A, stride_A * batch_size);
        curandGenerateUniform(gen, d_B, stride_B * batch_size);
        curandGenerateUniform(gen, d_C, stride_C * batch_size);
        
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Warmup
        for (int iter = 0; iter < warmup_iterations; iter++) {
            CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                                 m, n, k_dim,
                                                 &alpha,
                                                 d_A, m, stride_A,
                                                 d_B, k_dim, stride_B,
                                                 &beta,
                                                 d_C, m, stride_C,
                                                 batch_size));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        
        CHECK_CUDA(cudaEventRecord(start, 0));
        
        GmpProfiler::getInstance()->pushRange("StridedBatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                             CUBLAS_OP_N, CUBLAS_OP_N,
                                             m, n, k_dim,
                                             &alpha,
                                             d_A, m, stride_A,
                                             d_B, k_dim, stride_B,
                                             &beta,
                                             d_C, m, stride_C,
                                             batch_size));
        GmpProfiler::getInstance()->popRange("StridedBatchedGEMM", GmpProfileType::CONCURRENT_KERNEL);
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        
        float time_ms = 0;
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));
        
        // Performance metrics
        size_t total_memory_bytes = total_size_A + total_size_B + total_size_C;
        long long total_flops = batch_size * 2LL * m * n * k_dim;
        
        double throughput = calculateThroughput(total_memory_bytes, time_ms);
        double gflops = calculateGFLOPS(total_flops, time_ms);
        
        printHeader();
        printf("%-20s %-15.3f %-15.2f %-15.2f\n", 
               "Strided Batch GEMM", time_ms, throughput, gflops);
        
        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
        curandDestroyGenerator(gen);
        return true;
    }
    

    void printSystemInfo() {
        cudaDeviceProp prop;
        int device;
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
        
        printf("\n=== System Information ===\n");
        printf("Device: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("Memory Clock Rate: %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * prop.memoryClockRate * prop.memoryBusWidth / 8.0 / 1e6);
        printf("SM Count: %d\n", prop.multiProcessorCount);
        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Test Parameters: N=%d, k=%d\n", N, k);
        printf("Matrix dimensions - Big GEMM: %dx%d, Batched: %d x (%dx%d)\n", 
               N, N, k, N/k, N);
    }
    
    void runAllTests() {
        // printSystemInfo();
        // testBigGemm();
        // testNaiveGemm();
        // testBatchedGemm();
        // testStridedBatchedGemm();
        
        // printf("\n=== Performance Comparison Summary ===\n");
        // printf("Big GEMM tests single large %dx%d matrix multiplication\n", N, N);
        // printf("Naive GEMM tests %d individual %dx%d matrix multiplications using cublasSgemm\n", k, N/k, N);
        // printf("Batched GEMM tests %d separate %dx%d matrix multiplications using cublasSgemmBatched\n", k, N/k, N);
        // printf("Strided Batched GEMM tests %d %dx%d matrices in contiguous memory\n", k, N/k, N);
        // printf("\nNote: All tests use single precision (fp32) arithmetic with single timing measurement\n");


        printSystemInfo();
        
        printf("\n=== Running CUDA Performance Tests ===\n");
        printf("Note: If any test fails due to memory allocation or other errors, it will be skipped.\n\n");
        
        bool success;
        
        success = testNaiveGemm_v2(); 
        if (!success) printf("Naive GEMM test failed and was skipped.\n");
        
        success = testBatchedGemm_v2();
        if (!success) printf("Batched GEMM test failed and was skipped.\n");
        
        success = testStridedBatchedGemm_v2();
        if (!success) printf("Strided Batched GEMM test failed and was skipped.\n");
        
        printf("\nAll available tests completed.\n");
    }
};

int main(int argc, char **argv) {
    // Default parameters
    int N = 4096;  // Matrix dimension for big GEMM
    int k = 16;    // Number of batches
    int warmup_iter = 5;
    
    // Parse command line arguments
    if (argc >= 2) N = atoi(argv[1]);
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) warmup_iter = atoi(argv[3]);
    
    // Validate parameters
    if (N % k != 0) {
        printf("Error: N must be divisible by k. N=%d, k=%d\n", N, k);
        return 1;
    }
    
    printf("GEMM Performance Tester\n");
    printf("Usage: %s [N=%d] [k=%d] [warmup_iter=%d]\n", 
           argv[0], N, k, warmup_iter);
    printf("Where N is matrix size, k is batch count\n");

    GmpProfiler::getInstance()->init();

    // CUpti_Checkpoint handle{CUpti_Checkpoint_STRUCT_SIZE};
    // handle.optimizations = 1;
    // // Retain current context
    // CUcontext cuContext;
    // DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&cuContext, 0));
    // handle.ctx = cuContext;

    // NV::Cupti::Checkpoint::cuptiCheckpointSave(&handle);
    
    do{
        GemmPerformanceTester tester(N, k, warmup_iter);
        GmpProfiler::getInstance()->startRangeProfiling();
        tester.runAllTests();
        // NV::Cupti::Checkpoint::cuptiCheckpointRestore(&handle);
        GmpProfiler::getInstance()->stopRangeProfiling();
    }while(!GmpProfiler::getInstance()->hasSubmittedAllPasses());

    GmpProfiler::getInstance()->decodeCounterData();
    std::string name("N=" + std::to_string(N) + "_k=" + std::to_string(k));
    GmpProfiler::getInstance()->printProfilerRanges(name, GmpOutputKernelReduction::SUM);
    
    return 0;
}
