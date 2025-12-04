/*
 *  Copyright 2024 NVIDIA Corporation. All rights reserved
 */

#include <atomic>
#include <chrono>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <thread>
#include <chrono>

#ifdef _WIN32
#define strdup _strdup
#endif

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

#include "range_profiling.h"
#include "cupti_checkpoint.h"

// Kernels
__global__
void vectorAdd(const int *pA, const int *pB, int *pC, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        pC[i] = pA[i] + pB[i];
    }
}

class VectorLaunchWorkLoad
{
    int m_numOfElements;
    int m_threadsPerBlock, m_blocksPerGrid;
    size_t m_size;

    int *pDeviceA, *pDeviceB, *pDeviceC, *pDeviceD, *pDeviceE, *pDeviceF;
    std::vector<int> pHostA, pHostB, pHostC, pHostD, pHostE, pHostF;

public:
    VectorLaunchWorkLoad(int numElements = 8000000, int threadsPerBlock = 256) :
        m_numOfElements(numElements), m_threadsPerBlock(threadsPerBlock)
    {
        std::cout<<"VectorLaunchWorkLoad with "
                 << m_numOfElements << " elements and "
                 << m_threadsPerBlock << " threads per block." << std::endl;
        m_size = m_numOfElements * sizeof(int);
        m_blocksPerGrid = (m_numOfElements + m_threadsPerBlock - 1) / m_threadsPerBlock;
        pHostA.resize(m_numOfElements);
        pHostB.resize(m_numOfElements);
        pHostC.resize(m_numOfElements);
        pHostD.resize(m_numOfElements);
        pHostE.resize(m_numOfElements);
        pHostF.resize(m_numOfElements);
    }

    ~VectorLaunchWorkLoad() {}

    void InitializeVector(std::vector<int>& pVector)
    {
        for (int i = 0; i < m_numOfElements; i++) {
            pVector[i] = i;
        }
    }

    void CleanUp()
    {
        // Free device memory.
        RUNTIME_API_CALL(cudaFree(pDeviceA));
        RUNTIME_API_CALL(cudaFree(pDeviceB));
        RUNTIME_API_CALL(cudaFree(pDeviceC));
        RUNTIME_API_CALL(cudaFree(pDeviceD));
        RUNTIME_API_CALL(cudaFree(pDeviceE));
        RUNTIME_API_CALL(cudaFree(pDeviceF));
    }

    void SetUp()
    {
        // Initialize input vectors
        InitializeVector(pHostA);
        InitializeVector(pHostB);
        std::fill(pHostC.begin(), pHostC.end(), 0);

        // Allocate vectors in device memory
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceA, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceB, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceC, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceD, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceE, m_size));
        RUNTIME_API_CALL(cudaMalloc((void **)&pDeviceF, m_size));

        // Copy vectors from host memory to device memory
        // RUNTIME_API_CALL(cudaMemcpy(pDeviceA, pHostA.data(), m_size, cudaMemcpyHostToDevice));
        // RUNTIME_API_CALL(cudaMemcpy(pDeviceB, pHostB.data(), m_size, cudaMemcpyHostToDevice));
        // RUNTIME_API_CALL(cudaMemcpy(pDeviceD, pHostD.data(), m_size, cudaMemcpyHostToDevice));
        // RUNTIME_API_CALL(cudaMemcpy(pDeviceE, pHostE.data(), m_size, cudaMemcpyHostToDevice));
    }

    void TearDown()
    {
        // Kernel Launch Verification
        // Copy result from device memory to host memory
        // pHostC contains the result in host memory
        RUNTIME_API_CALL(cudaMemcpy(pHostC.data(), pDeviceC, m_size, cudaMemcpyDeviceToHost));

        // Verify result
        int sum;
        for (int i = 0; i < m_numOfElements; ++i)
        {
            sum  = pHostA[i] + pHostB[i];
            if (pHostC[i] != sum)
            {
                fprintf(stderr, "Error: Result verification failed.\n");
                exit(EXIT_FAILURE);
            }
        }
        printf("Result verification passed.\n");
        CleanUp();
    }

    void LaunchKernel()
    {
        vectorAdd<<<m_blocksPerGrid, m_threadsPerBlock>>>(pDeviceA, pDeviceB, pDeviceC, m_numOfElements);
    }

    void LaunchKernel2()
    {
        vectorAdd<<<m_blocksPerGrid, m_threadsPerBlock>>>(pDeviceD, pDeviceE, pDeviceF, m_numOfElements);
    }
};

struct ParsedArgs
{
    int deviceIndex = 0;
    std::string rangeMode = "auto";
    std::string replayMode = "user";
    uint64_t maxRange = 20;
    std::vector<const char*> metrics =
    {
    //   // Group 1
    //   "gpu__time_duration.sum",
    //   "gpu__time_duration.max",
    //   "gpc__cycles_elapsed.avg.per_second",
    //   "gpc__cycles_elapsed.max",
    //   "sm__cycles_active.max",

    //   // // Group 2
    //   // // Sub Group 1
    //   "smsp__inst_executed.sum",
    //   "smsp__sass_inst_executed_op_shared_ld.sum",
    //   "smsp__sass_inst_executed_op_shared_st.sum",
    //   "smsp__sass_inst_executed_op_global_ld.sum",
    //   "smsp__sass_inst_executed_op_global_st.sum",
    //   // // Sub Group 2
    //   "sm__pipe_alu_cycles_active.max",
    //   "sm__pipe_fma_cycles_active.max",
    //   "sm__pipe_tensor_cycles_active.max",
    //   "sm__pipe_shared_cycles_active.max",

    //   "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    //   "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    //   // // Sub Group 3
    //   "sm__sass_inst_executed_op_ldgsts_cache_access.sum",
    //   "sm__sass_inst_executed_op_ldgsts_cache_bypass.sum",

    //   // // Group 3
    //   // // Sub Group 1
    //   "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    //   // // Sub Group 2
    //   "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    //   // // Sub Group 3
    //   "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    //   // // Sub Group 4
    //   "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    //   // // Sub Group 5
    //   "sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_access.sum",
    //   "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum",
    //   "sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts_cache_bypass.sum",
    //   "sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_bypass.sum",
    //   // // Sub Group 6
    //   "lts__t_requests_srcunit_tex_op_read.sum",
    //   "lts__t_requests_srcunit_tex_op_write.sum",
    //   "dram__sectors_read.sum",
    //   "dram__sectors_write.sum",
    //   // // Sub Group 7
    //   "lts__t_requests_srcunit_l1_op_read.sum",
    //   "lts__t_requests_srcunit_l1_op_write.sum",

    //   // // Group 4
    //   // // Sub Group 1
    //   "smsp__average_warp_latency_per_inst_issued.ratio",
    //   // // Sub Group 2
    //   "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio",
    //   "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    //   // // Sub Group 3
    //   "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    //   "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    };
};

ParsedArgs parseArgs(int argc, char *argv[]);
void ProfilingDeviceSupportStatus(CUdevice device);

int main(int argc, char *argv[])
{
    ParsedArgs args = parseArgs(argc, argv);
    DRIVER_API_CALL(cuInit(0));

    printf("Starting Range Profiling\n");

    // Get the current ctx for the device
    CUdevice cuDevice;
    DRIVER_API_CALL(cuDeviceGet(&cuDevice, args.deviceIndex));
    ProfilingDeviceSupportStatus(cuDevice);

    int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    DRIVER_API_CALL(cuDeviceGetAttribute(&computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    printf("Compute Capability of Device: %d.%d\n", computeCapabilityMajor, computeCapabilityMinor);

    if (computeCapabilityMajor < 7 || (computeCapabilityMajor == 7 && computeCapabilityMinor < 5))
    {
        std::cerr << "Range Profiling is supported only on devices with compute capability 7.5 and above" << std::endl;
        exit(EXIT_FAILURE);
    }

    RangeProfilerConfig config;
    config.maxNumOfRanges = args.maxRange;
    config.minNestingLevel = 1;
    config.numOfNestingLevel =1;

    CuptiProfilerHostPtr pCuptiProfilerHost = std::make_shared<CuptiProfilerHost>();

    // Create a context
    CUcontext cuContext;
    DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));
    RangeProfilerTargetPtr pRangeProfilerTarget = std::make_shared<RangeProfilerTarget>(cuContext, config);

    // Get chip name
    std::string chipName;
    CUPTI_API_CALL(RangeProfilerTarget::GetChipName(cuDevice, chipName));

    // Get Counter availability image
    std::vector<uint8_t> counterAvailabilityImage;
    CUPTI_API_CALL(RangeProfilerTarget::GetCounterAvailabilityImage(cuContext, counterAvailabilityImage));

    // Create config image
    std::vector<uint8_t> configImage;
    pCuptiProfilerHost->SetUp(chipName, counterAvailabilityImage);
    CUPTI_API_CALL(pCuptiProfilerHost->CreateConfigImage(args.metrics, configImage));

    // Set up the workload
    VectorLaunchWorkLoad vectorLaunchWorkLoad;
    vectorLaunchWorkLoad.SetUp();

    // Enable Range profiler
    CUPTI_API_CALL(pRangeProfilerTarget->EnableRangeProfiler());

    // Create CounterData Image
    std::vector<uint8_t> counterDataImage;
    CUPTI_API_CALL(pRangeProfilerTarget->CreateCounterDataImage(args.metrics, counterDataImage));

    // Set range profiler configuration
    printf("Range Mode: %s\n", args.rangeMode.c_str());
    printf("Replay Mode: %s\n", args.replayMode.c_str());
    CUPTI_API_CALL(pRangeProfilerTarget->SetConfig(
        args.rangeMode == "auto" ? CUPTI_AutoRange : CUPTI_UserRange,
        args.replayMode == "kernel" ? CUPTI_KernelReplay : CUPTI_UserReplay,
        configImage,
        counterDataImage
    ));
    using NV::Cupti::Checkpoint::CUpti_Checkpoint;
    CUpti_Checkpoint handle{CUpti_Checkpoint_STRUCT_SIZE};
    handle.ctx = cuContext;
    handle.optimizations = 0;
    int passes = 0;
    do
    {
        if (passes == 0) {
            CUPTI_API_CALL(cuptiCheckpointSave(&handle));
        } else {
            CUPTI_API_CALL(cuptiCheckpointRestore(&handle));
        }
        std::cout << "Starting Pass: " << passes << std::endl;
        // Start Range Profiling
        CUPTI_API_CALL(pRangeProfilerTarget->StartRangeProfiler());
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Push Range (Level 1)
            CUPTI_API_CALL(pRangeProfilerTarget->PushRange("VectorAdd"));
            // CUPTI_API_CALL(pRangeProfilerTarget->PushRange("Nested VectorAdd1"));
            // // Launch CUDA workload
            vectorLaunchWorkLoad.LaunchKernel();
            cudaDeviceSynchronize();

            vectorLaunchWorkLoad.LaunchKernel2();
            cudaDeviceSynchronize();
            CUPTI_API_CALL(pRangeProfilerTarget->PopRange());

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::nano> duration = end - start;
            std::cout << "Host workload duration: " << duration.count() << " nanoseconds\n";
        }

        // Stop Range Profiling
        CUPTI_API_CALL(pRangeProfilerTarget->StopRangeProfiler());

        passes++;
    }
    while (!pRangeProfilerTarget->IsAllPassSubmitted());

    // Get Profiler Data
    CUPTI_API_CALL(pRangeProfilerTarget->DecodeCounterData());

    // Evaluate the results
    size_t numRanges = 0;
    CUPTI_API_CALL(pCuptiProfilerHost->GetNumOfRanges(counterDataImage, numRanges));
    for (size_t rangeIndex = 0; rangeIndex < numRanges; ++rangeIndex)
    {
        CUPTI_API_CALL(pCuptiProfilerHost->EvaluateCounterData(rangeIndex, args.metrics, counterDataImage));
    }

    pCuptiProfilerHost->PrintProfilerRanges();

    // Clean up
    CUPTI_API_CALL(pRangeProfilerTarget->DisableRangeProfiler());
    pCuptiProfilerHost->TearDown();
    vectorLaunchWorkLoad.TearDown();

    DRIVER_API_CALL(cuCtxDestroy(cuContext));
    return 0;
}

void PrintHelp()
{
    printf("Usage:\n");
    printf("  Range Profiling:\n");
    printf("    ./range_profiling [args]\n");
    printf("        --device/-d <deviceIndex> : Device index to run the range profiling\n");
    printf("        --range/-r <auto/user> : Range profiling mode. auto: ranges are defined around each kernel user: user defined ranges (Push/Pop API)\n");
    printf("        --replay/-e <kernel/user> : Replay mode needed for multi-pass metrics. kernel: replay will be done by CUPTI internally user: replay done explicitly by user\n");
    printf("        --maxNumRanges/-n <maximum number of ranges stored in counterdata> : Maximum number of ranges stored in counterdata\n");
    printf("        --metrics/-m <metric1,metric2,...> : List of metrics to be collected\n");
}

ParsedArgs parseArgs(int argc, char *argv[])
{
    ParsedArgs args;
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--device" || arg == "-d")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Missing value for argument: %s\n", arg.c_str());
                PrintHelp();
                exit(EXIT_FAILURE);
            }
            args.deviceIndex = std::stoi(argv[++i]);
        }
        else if (arg == "--range" || arg == "-r")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Missing value for argument: %s\n", arg.c_str());
                PrintHelp();
                exit(EXIT_FAILURE);
            }
            args.rangeMode = argv[++i];
        }
        else if (arg == "--replay" || arg == "-e")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Missing value for argument: %s\n", arg.c_str());
                PrintHelp();
                exit(EXIT_FAILURE);
            }
            args.replayMode = argv[++i];
        }
        else if (arg == "--maxNumRanges" || arg == "-n")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Missing value for argument: %s\n", arg.c_str());
                PrintHelp();
                exit(EXIT_FAILURE);
            }
            args.maxRange = std::stoull(argv[++i]);
        }
        else if (arg == "--metrics" || arg == "-m")
        {
            if (i + 1 >= argc)
            {
                fprintf(stderr, "Missing value for argument: %s\n", arg.c_str());
                PrintHelp();
                exit(EXIT_FAILURE);
            }
            std::stringstream ss(argv[++i]);
            std::string metric;
            args.metrics.clear();
            while (std::getline(ss, metric, ','))
            {
                args.metrics.push_back(strdup(metric.c_str()));
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            PrintHelp();
            exit(EXIT_SUCCESS);
        }
        else
        {
            fprintf(stderr, "Invalid argument: %s\n", arg.c_str());
            PrintHelp();
            exit(EXIT_FAILURE);
        }
    }
    return args;
}

void ProfilingDeviceSupportStatus(CUdevice device)
{
    CUpti_Profiler_DeviceSupported_Params params = { CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE };
    params.cuDevice = device;
    params.api = CUPTI_PROFILER_RANGE_PROFILING;
    CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

    if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED)
    {
        ::std::cerr << "Unable to profile on device " << device << ::std::endl;

        if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
        }

        if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice sli configuration is not supported" << ::std::endl;
        }

        if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice vgpu configuration is not supported" << ::std::endl;
        }
        else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED)
        {
            ::std::cerr << "\tdevice vgpu configuration disabled profiling support" << ::std::endl;
        }

        if (params.confidentialCompute == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tdevice confidential compute configuration is not supported" << ::std::endl;
        }

        if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported" << ::std::endl;
        }

        if (params.wsl == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED)
        {
            ::std::cerr << "\tWSL is not supported" << ::std::endl;
        }

        exit(EXIT_WAIVED);
    }
}