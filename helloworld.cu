#include <stdio.h>
#include <stdlib.h>
#include <cstring>
// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <iomanip>
#include <cassert>
#include "gmp/profile.h"
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cub/device/device_for.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cuda/std/mdspan>
#include <cuda/atomic>
#include <cupti_checkpoint.h>

#include <gmp/log.h>
#include <gmp/util.h>

// #define N 13824*4*1024*16 // vector length, 3.456GB
#define N (4<<20)
#define LARGE_N (4*128*768*32)

// #define CUPTI_CALL(call)                                                         \
//     do                                                                           \
//     {                                                                            \
//         CUptiResult _status = call;                                              \
//         if (_status != CUPTI_SUCCESS)                                            \
//         {                                                                        \
//             const char *errstr;                                                  \
//             cuptiGetResultString(_status, &errstr);                              \
//             fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
//                     __FILE__, __LINE__, #call, errstr);                          \
//             exit(-1);                                                            \
//         }                                                                        \
//     } while (0)

// Simple CUDA kernel
__global__ void hello_kernel()
{
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

__global__ void vecAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
}

__global__ void vecAdd_more_compute(int numIterations, float* sum)
{
    __shared__ float shared_A[10];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    for (size_t i = 0; i < numIterations; i++)
    {
        for (size_t j = 0; j < 10; j++)
        {
            sum[idx] += shared_A[j];
        }
    }
}

void vecAdd_thrust(float* out, float* inp1, float* inp2, int n) {
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp1cs(inp1);
    cub::CacheModifiedInputIterator<cub::LOAD_CS, float> inp2cs(inp2);
    thrust::transform(thrust::cuda::par_nosync, inp1cs, inp1cs + n, inp2cs, out, thrust::plus<float>());
}

__global__ void multiply(const float *A, const float *B, float *C, int numElements)
{
    for (int i =  blockDim.x * blockIdx.x + threadIdx.x; i < numElements; i += blockDim.x * gridDim.x)
        C[i] = A[i] * B[i];
}

__global__ void square(float *A, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        A[i] = A[i] * A[i];
    }
}

__global__ void saxpy(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}

__global__ void saxpy_more_compute(int n, float a, float *x, float *y)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < n; 
         i += blockDim.x * gridDim.x) 
      {
          y[i] = a * x[i] + y[i];
      }
}

__global__ void sumReduction(float *input, float *output, int n)
{
    __shared__ float sdata[256]; // shared memory for partial sums
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

// Test launch overhead by launching large number of kernels
// The total wall clock is compared with the total gpu time reported by GMP.
void testLaunchOverhead(float *d_A_1, float *d_B_1, float *d_C_1, float *d_A_2, float *d_B_2, float *d_C_2, float *d_A_3, float *d_B_3, float *d_C_3, float *d_A_4)
{
    GmpProfiler::getInstance()->pushRange("LaunchOverhead", GmpProfileType::CONCURRENT_KERNEL);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 110; i++)
    {
        multiply<<<108, 128>>>(d_A_1, d_B_1, d_C_1, N);
        multiply<<<108, 128>>>(d_A_3, d_B_3, d_C_3, N);
        multiply<<<108, 128>>>(d_A_2, d_B_2, d_C_2, N);
        square<<<108, 128>>>(d_A_4, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration = end - start;
    printf("Wall clock time for all kernel launches: %f ns\n",
           duration.count());
    GmpProfiler::getInstance()->popRange("LaunchOverhead", GmpProfileType::CONCURRENT_KERNEL);
}

void testComputeUtilization(float *d_A_1, float *d_B_1, float *d_C_1, float *d_A_2, float *d_B_2, float *d_C_2, float *d_A_3, float *d_B_3, float *d_C_3, float *d_A_4, float *d_B_4, float *d_C_4, float *d_A_5, float *d_B_5, float *d_C_5){
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    int num_sms = prop.multiProcessorCount;              // e.g., 108 on A100
    int smem_per_sm = prop.sharedMemPerMultiprocessor;   // bytes
    int max_smem_block_optin = 0;
    cudaDeviceGetAttribute(&max_smem_block_optin,
                        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);\

    
    // 2) Choose dynamic smem > half of per-SM shared mem so 2 blocks can’t co-reside
    int dyn_smem = std::min(max_smem_block_optin, smem_per_sm) / 2 + 4096; // +1KB cushion

    // 3) Opt-in to large per-block shared mem if needed (Ampere+)
    cudaFuncSetAttribute(vecAdd_more_compute,
    cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
    
    // 1/4 of sm_sp usage
    GmpProfiler::getInstance()->pushRange("ComputeUtilization_small_block_low_compute", GmpProfileType::CONCURRENT_KERNEL);
        vecAdd_more_compute<<<1, 1>>>(10000, d_A_1);
        vecAdd_more_compute<<<1, 32>>>(10000, d_A_1);
        vecAdd_more_compute<<<1, 128>>>(10000, d_A_1);
        vecAdd_more_compute<<<1, 1024>>>(10000, d_A_1);
        vecAdd_more_compute<<<54, 32>>>(10000, d_A_1);
        vecAdd_more_compute<<<108, 1>>>(10000, d_A_1);
        vecAdd_more_compute<<<54, 1024>>>(10000, d_A_1);
        vecAdd_more_compute<<<108, 1024>>>(10000, d_A_1);
        vecAdd_more_compute<<<532, 1024>>>(10000, d_A_1);
    GmpProfiler::getInstance()->popRange("ComputeUtilization_small_block_low_compute", GmpProfileType::CONCURRENT_KERNEL);

    // // boost num instructions
    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_small_block_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms, 32>>>(d_A_2, d_B_2, d_C_2, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_small_block_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    // // 100% of sm_sp usage
    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_full_block_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms, 128>>>(d_A_3, d_B_3, d_C_3, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_full_block_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    // // 200% of sm_sp usage
    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_big_block_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms, 256>>>(d_A_4, d_B_4, d_C_4, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_big_block_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    //     // 400% of sm_sp usage
    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_big_block_4x_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms, 512>>>(d_A_5, d_B_5, d_C_5, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_big_block_4x_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_big_block_8x_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms*2, 512>>>(d_A_5, d_B_5, d_C_5, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_big_block_8x_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    //     GmpProfiler::getInstance()->pushRange("ComputeUtilization_big_block_16x_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms*4, 512>>>(d_A_5, d_B_5, d_C_5, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_big_block_16x_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_big_block_32x_high_compute", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms*6, 512>>>(d_A_5, d_B_5, d_C_5, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_big_block_32x_high_compute", GmpProfileType::CONCURRENT_KERNEL);

    // GmpProfiler::getInstance()->pushRange("ComputeUtilization_blocksize", GmpProfileType::CONCURRENT_KERNEL);
    //     vecAdd_more_compute<<<num_sms*6, 512*2>>>(d_A_5, d_B_5, d_C_5, N, 1000);
    // GmpProfiler::getInstance()->popRange("ComputeUtilization_blocksize", GmpProfileType::CONCURRENT_KERNEL);
}

// This function reveals the bandwidth distribution among multiple kernels launched within a logical range.
// The bandwidth reported by GMP has a high-low-high pattern.
void testBandwidthDist(float *d_A_5, float *d_B_5, float *d_C_5)
{
    GmpProfiler::getInstance()->pushRange("BandwidthDist", GmpProfileType::CONCURRENT_KERNEL);
    for(int i = 0; i<20;i++){
        // Typical workflow in a range
        // Each kernel accept the ouput of previous kernel as input.
        saxpy<<<108, 128>>>(N, 2.0f, d_A_5, d_B_5);
        multiply<<<108, 128>>>(d_A_5, d_B_5, d_C_5, N);
        multiply<<<108, 128>>>(d_A_5, d_B_5, d_C_5, N);
        square<<<108, 128>>>(d_C_5, N);
    }
    
    GmpProfiler::getInstance()->popRange("BandwidthDist", GmpProfileType::CONCURRENT_KERNEL);
}
using NV::Cupti::Checkpoint::CUpti_Checkpoint;
using NV::Cupti::Checkpoint::cuptiCheckpointSave;
using NV::Cupti::Checkpoint::cuptiCheckpointRestore;
void launchKernelWithRestore(float *d_A, float *d_B, float *d_C, CUpti_Checkpoint &handle, int gridSize, int blockSize)
{
    vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cuptiCheckpointRestore(&handle);
}

void testOccupancy(float *d_A, float *d_B, float *d_C)
{
    CUpti_Checkpoint handle{CUpti_Checkpoint_STRUCT_SIZE};
    handle.optimizations = 1;
    // Retain current context
    CUcontext cuContext;
    DRIVER_API_CALL(cuDevicePrimaryCtxRetain(&cuContext, 0));
    handle.ctx = cuContext;

    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,1);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,64);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,128);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,256);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,512);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,1024);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 1,2048);

    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,1);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,64);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,128);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,256);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,512);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,1024);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 108,2048);

    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,1);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,64);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,128);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,256);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,512);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,1024);
    launchKernelWithRestore(d_A, d_B, d_C, handle, 216,2048);
}

void launch_kernels()
{
    size_t size = N * sizeof(float);
    size_t large_size = LARGE_N * sizeof(float);

    // No need to initialize the input data, as we are not copying from host to device
    // If N is too large, these host arrays lead to segment fault

    // Host vectors
    // float h_A[N], h_B[N], h_C[N];
    // for (int i = 0; i < N; i++)
    // {
    //     h_A[i] = i;
    //     h_B[i] = i * 10;
    // }

    // Device vectors
    float *d_A_1, *d_B_1, *d_C_1;
    float *d_A_2, *d_B_2, *d_C_2;
    float *d_A_3, *d_B_3, *d_C_3;
    float *d_A_4, *d_B_4, *d_C_4;
    float *d_A_5, *d_B_5, *d_C_5;   // Large arraies

    cudaMalloc((void **)&d_A_1, size);
    cudaMalloc((void **)&d_B_1, size);
    cudaMalloc((void **)&d_C_1, size);

    cudaMalloc((void **)&d_A_2, size);
    cudaMalloc((void **)&d_B_2, size);
    cudaMalloc((void **)&d_C_2, size);

    cudaMalloc((void **)&d_A_3, size);
    cudaMalloc((void **)&d_B_3, size);
    cudaMalloc((void **)&d_C_3, size);

    cudaMalloc((void **)&d_A_4, size);
    cudaMalloc((void **)&d_B_4, size);
    cudaMalloc((void **)&d_C_4, size);

    cudaMalloc((void **)&d_A_5, size);
    cudaMalloc((void **)&d_B_5, size);
    cudaMalloc((void **)&d_C_5, size);
    printf("Allocated device memory\n");

    testOccupancy(d_A_1, d_B_1, d_C_1);

    // Cleanup
    cudaFree(d_A_1);
    cudaFree(d_B_1);
    cudaFree(d_C_1);
    cudaFree(d_A_2);
    cudaFree(d_B_2);
    cudaFree(d_C_2);
    cudaFree(d_A_3);
    cudaFree(d_B_3);
    cudaFree(d_C_3);
}

int main(int argc, char **argv)
{
  std::string outputPath = "";
  GmpOutputKernelReduction outputOption = GmpOutputKernelReduction::SUM;
  assert(argc >= 2);
  for(int i = 2; i < argc; i++)
  {
      if(strcmp(argv[i], "-o") == 0){
          printf("Setting output path: %s\n", argv[i + 1]);
          assert(i + 1 < argc);
          outputPath = argv[i + 1];
          i++;
      }
      else if(strcmp(argv[i], "--max") == 0){
          outputOption = GmpOutputKernelReduction::MAX;
          printf("Setting output option to MAX\n");
      }
      else if(strcmp(argv[i], "--mean") == 0){
          outputOption = GmpOutputKernelReduction::MEAN;
          printf("Setting output option to MEAN\n");
      }
      else if(strcmp(argv[i], "--sum") == 0){
          outputOption = GmpOutputKernelReduction::SUM;
          printf("Setting output option to SUM\n");
      }
      else{
          printf("Adding metric: %s\n", argv[i]);
          GmpProfiler::getInstance()->addMetrics(argv[i]);
      }  
  }

    // hello_kernel<<<1, 4>>>();
    int curr_pass = 0;
    GmpProfiler::getInstance()->init();

    printf("Starting profiling runs...\n");
    printf("current pass: %zu\n", curr_pass++);
    GmpProfiler::getInstance()->startRangeProfiling();
    launch_kernels();
    GmpProfiler::getInstance()->stopRangeProfiling();

    cudaDeviceSynchronize();
    GmpProfiler::getInstance()->decodeCounterData();
    GmpProfiler::getInstance()->printProfilerRanges(outputOption);
    GmpProfiler::getInstance()->produceOutput(outputOption);

    // CUPTI_CALL(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    return 0;
}
