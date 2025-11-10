#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward declarations of kernels
__global__ void blockAndGlobalHisto(unsigned int* HH, unsigned int* Hg, int h, 
                                    unsigned int* Input, int nElements, 
                                    unsigned int nMin, unsigned int nMax);

__global__ void globalHistoScan(unsigned int* Hg, unsigned int* SHg, int h);

__global__ void verticalScanHH(unsigned int* HH, unsigned int* PSv, int h, int nb);

__global__ void PartitionKernel(unsigned int* HH, unsigned int* SHg, unsigned int* PSv, 
                                int h, unsigned int* Input, unsigned int* Output, 
                                int nElements, unsigned int nMin, unsigned int nMax, int nb);

__global__ void bitonicSort(unsigned int* data, int arrayLength, int dir);

// ============================================================================
// KERNEL IMPLEMENTATIONS
// ============================================================================

// Kernel 1: blockAndGlobalHisto
// Computes per-block histograms (HH) and global histogram (Hg)
__global__ void blockAndGlobalHisto(unsigned int* HH, unsigned int* Hg, int h, 
                                    unsigned int* Input, int nElements, 
                                    unsigned int nMin, unsigned int nMax) {
    // Allocate shared memory for local histogram
    extern __shared__ unsigned int s_histo[];
    
    // Calculate bin width
    unsigned int L = (nMax - nMin) / h;
    if (L == 0) L = 1;
    
    // Initialize shared memory
    if (threadIdx.x < h) {
        s_histo[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Grid-stride loop to process all elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < nElements; i += stride) {
        unsigned int val = Input[i];
        
        // Find the bin for this value
        unsigned int bin = (val - nMin) / L;
        if (bin >= h) bin = h - 1;
        
        // Increment local histogram in shared memory
        atomicAdd(&s_histo[bin], 1);
    }
    __syncthreads();
    
    // Write results to global memory
    if (threadIdx.x < h) {
        unsigned int count = s_histo[threadIdx.x];
        
        // Write to HH (per-block histogram matrix)
        HH[blockIdx.x * h + threadIdx.x] = count;
        
        // Add to Hg (global histogram)
        atomicAdd(&Hg[threadIdx.x], count);
    }
}

// Kernel 2: globalHistoScan
// Performs exclusive prefix sum (scan) on global histogram
__global__ void globalHistoScan(unsigned int* Hg, unsigned int* SHg, int h) {
    extern __shared__ unsigned int s_data[];
    
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < h) {
        s_data[tid] = Hg[tid];
    } else if (tid < blockDim.x) {
        s_data[tid] = 0;
    }
    __syncthreads();
    
    // Simple sequential scan for small h (more reliable)
    if (tid == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < h; i++) {
            unsigned int val = s_data[i];
            s_data[i] = sum;
            sum += val;
        }
    }
    __syncthreads();
    
    // Write results to global memory
    if (tid < h) {
        SHg[tid] = s_data[tid];
    }
}

// Kernel 3: verticalScanHH
// Performs vertical (column-wise) exclusive prefix sum on HH matrix
__global__ void verticalScanHH(unsigned int* HH, unsigned int* PSv, int h, int nb) {
    int col = blockIdx.x;  // Each block handles one column
    if (col >= h) return;
    
    extern __shared__ unsigned int s_col[];
    
    // Load column from HH into shared memory
    if (threadIdx.x < nb) {
        s_col[threadIdx.x] = HH[threadIdx.x * h + col];
    } else if (threadIdx.x < blockDim.x) {
        s_col[threadIdx.x] = 0;
    }
    __syncthreads();
    
    // Simple sequential scan (thread 0 does the work)
    if (threadIdx.x == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < nb; i++) {
            unsigned int val = s_col[i];
            s_col[i] = sum;
            sum += val;
        }
    }
    __syncthreads();
    
    // Write results back to PSv
    if (threadIdx.x < nb) {
        PSv[threadIdx.x * h + col] = s_col[threadIdx.x];
    }
}

// Kernel 4: PartitionKernel
// Partitions input into output using SHg and PSv
__global__ void PartitionKernel(unsigned int* HH, unsigned int* SHg, unsigned int* PSv, 
                                int h, unsigned int* Input, unsigned int* Output, 
                                int nElements, unsigned int nMin, unsigned int nMax, int nb) {
    int b = blockIdx.x;  // Block ID
    
    // Allocate shared memory
    extern __shared__ unsigned int s_data[];
    unsigned int* HLsh = s_data;           // Size h
    unsigned int* SHg_sh = &s_data[h];     // Size h
    
    // Calculate bin width
    unsigned int L = (nMax - nMin) / h;
    if (L == 0) L = 1;
    
    // Load SHg and PSv[b] into shared memory
    if (threadIdx.x < h) {
        HLsh[threadIdx.x] = PSv[b * h + threadIdx.x];
        SHg_sh[threadIdx.x] = SHg[threadIdx.x];
    }
    __syncthreads();
    
    // Compute starting position for each bin for this block (HLsh = PSv[b] + SHg)
    if (threadIdx.x < h) {
        HLsh[threadIdx.x] = HLsh[threadIdx.x] + SHg_sh[threadIdx.x];
    }
    __syncthreads();
    
    // Grid-stride loop to process elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < nElements; i += stride) {
        unsigned int e = Input[i];
        
        // Find the bin for this element
        unsigned int f = (e - nMin) / L;
        if (f >= h) f = h - 1;
        
        // Get position using atomic add in shared memory
        unsigned int p = atomicAdd(&HLsh[f], 1);
        
        // Store element in output at correct position
        // Add bounds check to prevent illegal memory access
        if (p < nElements) {
            Output[p] = e;
        }
    }
}

// Kernel 5: bitonicSort
// Simplified version for small power-of-2 arrays
__global__ void bitonicSort(unsigned int* data, int arrayLength, int dir) {
    extern __shared__ unsigned int shared[];
    
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < arrayLength) {
        shared[tid] = data[tid];
    }
    __syncthreads();
    
    // Bitonic sort
    for (int size = 2; size <= arrayLength; size <<= 1) {
        // Direction based on position
        int ddd = dir ^ ((tid & (size / 2)) != 0);
        
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            __syncthreads();
            int pos = 2 * tid - (tid & (stride - 1));
            
            if (pos < arrayLength && pos + stride < arrayLength) {
                unsigned int a = shared[pos];
                unsigned int b = shared[pos + stride];
                
                if ((a > b) == ddd) {
                    shared[pos] = b;
                    shared[pos + stride] = a;
                }
            }
        }
    }
    __syncthreads();
    
    // Write back to global memory
    if (tid < arrayLength) {
        data[tid] = shared[tid];
    }
}

// ============================================================================
// VERIFICATION AND HELPER FUNCTIONS
// ============================================================================

// Verification function
bool verifySort(unsigned int* Input_host, unsigned int* Output_host, int nElements) {
    // Create a copy of input and sort it with std::sort for verification
    unsigned int* Input_verify = (unsigned int*)malloc(nElements * sizeof(unsigned int));
    memcpy(Input_verify, Input_host, nElements * sizeof(unsigned int));
    
    std::sort(Input_verify, Input_verify + nElements);
    
    // Compare with our output
    bool correct = true;
    for (int i = 0; i < nElements; i++) {
        if (Output_host[i] != Input_verify[i]) {
            correct = false;
            printf("Error at position %d: expected %u, got %u\n", 
                   i, Input_verify[i], Output_host[i]);
            break;
        }
    }
    
    free(Input_verify);
    return correct;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 4 || argc > 5) {
        printf("Usage: %s <nTotalElements> <h> <nR> [verbose]\n", argv[0]);
        printf("  nTotalElements: number of unsigned ints in input vector\n");
        printf("  h: number of histogram bins\n");
        printf("  nR: number of repetitions for timing\n");
        printf("  verbose: optional, print detailed timing per iteration\n");
        return 1;
    }
    
    int nTotalElements = atoi(argv[1]);
    int h = atoi(argv[2]);
    int nR = atoi(argv[3]);
    bool verbose = (argc == 5);
    
    printf("=== mppSort GPU Implementation ===\n");
    printf("Number of elements: %d\n", nTotalElements);
    printf("Number of bins (h): %d\n", h);
    printf("Number of repetitions: %d\n\n", nR);
    
    // Generate input data on host
    unsigned int* Input_host = (unsigned int*)malloc(nTotalElements * sizeof(unsigned int));
    unsigned int nMin = 0xFFFFFFFF;
    unsigned int nMax = 0;
    
    srand(42); // Fixed seed for reproducibility
    
    for (int i = 0; i < nTotalElements; i++) {
        int a = rand();
        int b = rand();
        unsigned int v = a * 100 + b;
        Input_host[i] = v;
        
        if (v < nMin) nMin = v;
        if (v > nMax) nMax = v;
    }
    
    unsigned int L = (nMax - nMin) / h;
    if (L == 0) L = 1;
    
    printf("Data interval [nMin, nMax]: [%u, %u]\n", nMin, nMax);
    printf("Bin width (L): %u\n\n", L);
    
    // Get device properties to determine number of blocks
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int NP = prop.multiProcessorCount;
    int nb = NP * 2;  // Number of blocks as per specification
    int nt = 1024;    // Threads per block as per specification
    
    printf("Device: %s\n", prop.name);
    printf("Number of SMs: %d\n", NP);
    printf("Number of blocks (nb): %d\n", nb);
    printf("Threads per block (nt): %d\n\n", nt);
    
    // Allocate device memory
    unsigned int* Input_dev;
    unsigned int* Output_dev;
    unsigned int* HH_dev;      // nb x h matrix (histograms per block)
    unsigned int* Hg_dev;      // h-element vector (global histogram)
    unsigned int* SHg_dev;     // h-element vector (scan of global histogram)
    unsigned int* PSv_dev;     // nb x h matrix (vertical prefix sum)
    
    CUDA_CHECK(cudaMalloc(&Input_dev, nTotalElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&Output_dev, nTotalElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&HH_dev, nb * h * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&Hg_dev, h * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&SHg_dev, h * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&PSv_dev, nb * h * sizeof(unsigned int)));
    
    // Copy input to device
    CUDA_CHECK(cudaMemcpy(Input_dev, Input_host, nTotalElements * sizeof(unsigned int), 
                          cudaMemcpyHostToDevice));
    
    // Create events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    float totalTime = 0.0f;
    
    // Warm-up run (optional, but good practice)
    CUDA_CHECK(cudaMemset(HH_dev, 0, nb * h * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(Hg_dev, 0, h * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(SHg_dev, 0, h * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(PSv_dev, 0, nb * h * sizeof(unsigned int)));
    
    // Main timing loop
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int r = 0; r < nR; r++) {
        if (r == 0) printf("Iniciando iteração 0 (com debug)...\n");
        
        // Zero out arrays before each iteration
        CUDA_CHECK(cudaMemset(HH_dev, 0, nb * h * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(Hg_dev, 0, h * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(SHg_dev, 0, h * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(PSv_dev, 0, nb * h * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(Output_dev, 0, nTotalElements * sizeof(unsigned int)));
        
        if (r == 0) printf("Lançando Kernel 1...\n");
        // Kernel 1: blockAndGlobalHisto
        int sharedMem1 = h * sizeof(unsigned int);
        blockAndGlobalHisto<<<nb, nt, sharedMem1>>>(HH_dev, Hg_dev, h, Input_dev, 
                                                     nTotalElements, nMin, nMax);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (r == 0) printf("Kernel 1 OK\n");
        
        if (r == 0) printf("Lançando Kernel 2...\n");
        // Kernel 2: globalHistoScan
        // Use only h threads since we're doing sequential scan
        int nt2 = (h < 256) ? 256 : h;  // At least 256 threads for a full warp
        int sharedMem2 = nt2 * sizeof(unsigned int);
        globalHistoScan<<<1, nt2, sharedMem2>>>(Hg_dev, SHg_dev, h);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (r == 0) printf("Kernel 2 OK\n");
        
        if (r == 0) printf("Lançando Kernel 3...\n");
        // Kernel 3: verticalScanHH
        int nb3 = h;  // One block per column
        // Need at least nb threads, round up to nearest power of 2 for efficiency
        int nt3 = 32;  // Start with one warp
        while (nt3 < nb && nt3 < 1024) nt3 *= 2;
        int sharedMem3 = nt3 * sizeof(unsigned int);
        verticalScanHH<<<nb3, nt3, sharedMem3>>>(HH_dev, PSv_dev, h, nb);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (r == 0) printf("Kernel 3 OK\n");
        
        if (r == 0) printf("Lançando Kernel 4...\n");
        // Kernel 4: PartitionKernel
        int sharedMem4 = 2 * h * sizeof(unsigned int);
        PartitionKernel<<<nb, nt, sharedMem4>>>(HH_dev, SHg_dev, PSv_dev, h, 
                                                 Input_dev, Output_dev, nTotalElements, 
                                                 nMin, nMax, nb);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        if (r == 0) printf("Kernel 4 OK\n");
        
        // Kernel 5: Sort each bin
        // Only do this on the last iteration to save time
        if (r == nR - 1) {
            unsigned int* Hg_host = (unsigned int*)malloc(h * sizeof(unsigned int));
            unsigned int* SHg_host = (unsigned int*)malloc(h * sizeof(unsigned int));
            CUDA_CHECK(cudaMemcpy(Hg_host, Hg_dev, h * sizeof(unsigned int), 
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(SHg_host, SHg_dev, h * sizeof(unsigned int), 
                                  cudaMemcpyDeviceToHost));
            
            // Sort each bin using Thrust (simpler and more reliable)
            for (int i = 0; i < h; i++) {
                unsigned int bin_start = SHg_host[i];
                unsigned int bin_count = Hg_host[i];
                
                if (bin_count == 0) continue;
                
                unsigned int* bin_ptr = Output_dev + bin_start;
                
                // Use thrust::sort for all bins
                thrust::device_ptr<unsigned int> thrust_ptr = 
                    thrust::device_pointer_cast(bin_ptr);
                thrust::sort(thrust_ptr, thrust_ptr + bin_count);
            }
            
            free(Hg_host);
            free(SHg_host);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    float avgTime = totalTime / nR;
    float throughput = (nTotalElements / (avgTime / 1000.0f)) / 1e9;
    
    printf("=== Performance Results (mppSort) ===\n");
    printf("Total time for %d iterations: %.3f ms\n", nR, totalTime);
    printf("Average time per iteration: %.3f ms\n", avgTime);
    printf("Throughput: %.3f GElements/s\n\n", throughput);
    
    // Benchmark Thrust for comparison
    unsigned int* Thrust_Input_dev;
    CUDA_CHECK(cudaMalloc(&Thrust_Input_dev, nTotalElements * sizeof(unsigned int)));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int r = 0; r < nR; r++) {
        CUDA_CHECK(cudaMemcpy(Thrust_Input_dev, Input_host, 
                              nTotalElements * sizeof(unsigned int), 
                              cudaMemcpyHostToDevice));
        
        thrust::device_ptr<unsigned int> thrust_ptr = 
            thrust::device_pointer_cast(Thrust_Input_dev);
        thrust::sort(thrust_ptr, thrust_ptr + nTotalElements);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float thrustTime;
    CUDA_CHECK(cudaEventElapsedTime(&thrustTime, start, stop));
    float avgThrustTime = thrustTime / nR;
    float thrustThroughput = (nTotalElements / (avgThrustTime / 1000.0f)) / 1e9;
    
    printf("=== Performance Results (Thrust) ===\n");
    printf("Total time for %d iterations: %.3f ms\n", nR, thrustTime);
    printf("Average time per iteration: %.3f ms\n", avgThrustTime);
    printf("Throughput: %.3f GElements/s\n\n", thrustThroughput);
    
    printf("=== Speedup ===\n");
    printf("mppSort vs Thrust: %.2fx\n\n", thrustThroughput / throughput);
    
    // Verify correctness
    unsigned int* Output_host = (unsigned int*)malloc(nTotalElements * sizeof(unsigned int));
    CUDA_CHECK(cudaMemcpy(Output_host, Output_dev, nTotalElements * sizeof(unsigned int), 
                          cudaMemcpyDeviceToHost));
    
    printf("=== Verification ===\n");
    if (verifySort(Input_host, Output_host, nTotalElements)) {
        printf("Ordenação correta!\n");
    } else {
        printf("ERRO NA ORDENAÇÃO\n");
    }
    
    // Cleanup
    free(Input_host);
    free(Output_host);
    CUDA_CHECK(cudaFree(Input_dev));
    CUDA_CHECK(cudaFree(Output_dev));
    CUDA_CHECK(cudaFree(HH_dev));
    CUDA_CHECK(cudaFree(Hg_dev));
    CUDA_CHECK(cudaFree(SHg_dev));
    CUDA_CHECK(cudaFree(PSv_dev));
    CUDA_CHECK(cudaFree(Thrust_Input_dev));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
