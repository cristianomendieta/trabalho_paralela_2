/*
 * mppSort - Parallel Sorting Algorithm for GPU
 * Trabalho 3 - CI1009 Programação Paralela com GPUs
 * 
 * Implementação da Parte B: mppSort com Segmented Bitonic Sort
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm>
#include <vector>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

typedef unsigned int uint;

// ============================================================================
// PART A KERNEL (Segmented Bitonic Sort)
// ============================================================================

inline __device__ void Comparator(
    uint &keyA,
    uint &keyB,
    uint dir
) {
    uint t;
    if ((keyA > keyB) == dir) {
        t = keyA;
        keyA = keyB;
        keyB = t;
    }
}

__global__ void segmentedBitonicSortKernel(
    uint *d_DstKey,
    uint *d_SrcKey,
    uint *d_Offsets,
    uint *d_Sizes,
    uint dir
) {
    extern __shared__ uint s_key[];

    uint seg_idx = blockIdx.x;
    uint offset = d_Offsets[seg_idx];
    uint arrayLength = d_Sizes[seg_idx];

    if (arrayLength == 0) return;

    uint padded_size = (arrayLength == 1) ? 1 : (1 << (32 - __clz(arrayLength - 1)));
    uint pad_value = dir ? UINT_MAX : 0;

    uint tid = threadIdx.x;
    for (uint i = tid; i < padded_size; i += blockDim.x) {
        if (i < arrayLength)
            s_key[i] = d_SrcKey[offset + i];
        else
            s_key[i] = pad_value;
    }
    __syncthreads();

    for (uint k = 2; k < padded_size; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < padded_size; i += blockDim.x) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    Comparator(s_key[i], s_key[ixj], (i & k) == 0);
                }
            }
            __syncthreads();
        }
    }

    for (uint j = padded_size >> 1; j > 0; j >>= 1) {
        for (uint i = tid; i < padded_size; i += blockDim.x) {
            uint ixj = i ^ j;
            if (ixj > i) {
                Comparator(s_key[i], s_key[ixj], dir);
            }
        }
        __syncthreads();
    }

    for (uint i = tid; i < arrayLength; i += blockDim.x) {
        d_DstKey[offset + i] = s_key[i];
    }
}

// ============================================================================
// PART B KERNELS (mppSort)
// ============================================================================

__global__ void blockAndGlobalHisto(uint* HH, uint* Hg, int h, 
                                    uint* Input, int nElements, 
                                    uint nMin, uint nMax) {
    extern __shared__ uint s_histo[];
    uint L = (nMax - nMin) / h;
    if (L == 0) L = 1;
    
    if (threadIdx.x < h) s_histo[threadIdx.x] = 0;
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < nElements; i += stride) {
        uint val = Input[i];
        uint bin = (val - nMin) / L;
        if (bin >= h) bin = h - 1;
        atomicAdd(&s_histo[bin], 1);
    }
    __syncthreads();
    
    if (threadIdx.x < h) {
        uint count = s_histo[threadIdx.x];
        HH[blockIdx.x * h + threadIdx.x] = count;
        atomicAdd(&Hg[threadIdx.x], count);
    }
}

__global__ void globalHistoScan(uint* Hg, uint* SHg, int h) {
    extern __shared__ uint s_data[];
    int tid = threadIdx.x;
    if (tid < h) s_data[tid] = Hg[tid];
    __syncthreads();
    
    if (tid == 0) {
        uint sum = 0;
        for (int i = 0; i < h; i++) {
            SHg[i] = sum;
            sum += s_data[i];
        }
    }
}

__global__ void verticalScanHH(uint* HH, uint* PSv, int h, int nb) {
    int c = blockIdx.x;
    if (c >= h) return;
    
    if (threadIdx.x == 0) {
        uint sum = 0;
        for (int r = 0; r < nb; r++) {
            PSv[r * h + c] = sum;
            sum += HH[r * h + c];
        }
    }
}

__global__ void PartitionKernel(uint* HH, uint* SHg, uint* PSv, 
                                int h, uint* Input, uint* Output, 
                                int nElements, uint nMin, uint nMax, int nb) {
    int b = blockIdx.x;
    extern __shared__ uint s_data[];
    uint* HLsh = s_data;           // Size h
    uint* SHg_sh = &s_data[h];     // Size h
    
    uint L = (nMax - nMin) / h;
    if (L == 0) L = 1;
    
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        SHg_sh[i] = SHg[i];
        HLsh[i] = PSv[b * h + i] + SHg_sh[i];
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < nElements; i += stride) {
        uint e = Input[i];
        uint f = (e - nMin) / L;
        if (f >= h) f = h - 1;
        
        uint p = atomicAdd(&HLsh[f], 1);
        if (p < nElements) {
            Output[p] = e;
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

bool verifySort(uint* Input_host, uint* Output_host, int nElements) {
    std::vector<uint> sorted(Input_host, Input_host + nElements);
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < nElements; i++) {
        if (Output_host[i] != sorted[i]) {
            printf("Mismatch at %d: expected %u, got %u\n", i, sorted[i], Output_host[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <nTotalElements> <h> <nR>\n", argv[0]);
        return 1;
    }
    
    int nTotalElements = atoi(argv[1]);
    int h = atoi(argv[2]);
    int nR = atoi(argv[3]);
    
    printf("n=%d, h=%d, nR=%d\n", nTotalElements, h, nR);
    
    // Host memory
    uint* h_Input = (uint*)malloc(nTotalElements * sizeof(uint));
    uint* h_Output = (uint*)malloc(nTotalElements * sizeof(uint));
    
    // Initialize
    srand(42);
    uint nMin = 0xFFFFFFFF, nMax = 0;
    for (int i = 0; i < nTotalElements; i++) {
        h_Input[i] = rand(); 
        if (h_Input[i] < nMin) nMin = h_Input[i];
        if (h_Input[i] > nMax) nMax = h_Input[i];
    }
    
    // Device memory
    uint *d_Input, *d_Output, *d_HH, *d_Hg, *d_SHg, *d_PSv;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int nb = prop.multiProcessorCount * 2;
    int nt = 1024;
    
    CUDA_CHECK(cudaMalloc(&d_Input, nTotalElements * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_Output, nTotalElements * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_HH, nb * h * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_Hg, h * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_SHg, h * sizeof(uint)));
    CUDA_CHECK(cudaMalloc(&d_PSv, nb * h * sizeof(uint)));
    
    CUDA_CHECK(cudaMemcpy(d_Input, h_Input, nTotalElements * sizeof(uint), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float totalTime = 0;
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < nR; r++) {
        CUDA_CHECK(cudaMemset(d_HH, 0, nb * h * sizeof(uint)));
        CUDA_CHECK(cudaMemset(d_Hg, 0, h * sizeof(uint)));
        CUDA_CHECK(cudaMemset(d_SHg, 0, h * sizeof(uint)));
        CUDA_CHECK(cudaMemset(d_PSv, 0, nb * h * sizeof(uint)));
        
        blockAndGlobalHisto<<<nb, nt, h * sizeof(uint)>>>(d_HH, d_Hg, h, d_Input, nTotalElements, nMin, nMax);
        globalHistoScan<<<1, 1, h * sizeof(uint)>>>(d_Hg, d_SHg, h); 
        verticalScanHH<<<h, 32>>>(d_HH, d_PSv, h, nb); 
        PartitionKernel<<<nb, nt, 2 * h * sizeof(uint)>>>(d_HH, d_SHg, d_PSv, h, d_Input, d_Output, nTotalElements, nMin, nMax, nb);
        
        // Segmented Sort
        segmentedBitonicSortKernel<<<h, 1024, 48 * 1024>>>(d_Output, d_Output, d_SHg, d_Hg, 1);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    
    printf("mppSort Time: %.3f ms\n", totalTime / nR);
    double throughput = (double)nTotalElements / (totalTime / nR / 1000.0) / 1e9;
    printf("Throughput: %.3f GEls/s\n", throughput);
    
    // Verification
    CUDA_CHECK(cudaMemcpy(h_Output, d_Output, nTotalElements * sizeof(uint), cudaMemcpyDeviceToHost));
    if (verifySort(h_Input, h_Output, nTotalElements)) {
        printf("Verification: PASS\n");
    } else {
        printf("Verification: FAIL\n");
    }
    
    // Thrust Benchmark
    uint* d_ThrustInput;
    CUDA_CHECK(cudaMalloc(&d_ThrustInput, nTotalElements * sizeof(uint)));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < nR; r++) {
        CUDA_CHECK(cudaMemcpy(d_ThrustInput, h_Input, nTotalElements * sizeof(uint), cudaMemcpyHostToDevice));
        thrust::sort(thrust::device_ptr<uint>(d_ThrustInput), thrust::device_ptr<uint>(d_ThrustInput + nTotalElements));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float thrustTime;
    CUDA_CHECK(cudaEventElapsedTime(&thrustTime, start, stop));
    
    printf("Thrust Time: %.3f ms\n", thrustTime / nR);
    double thrustThroughput = (double)nTotalElements / (thrustTime / nR / 1000.0) / 1e9;
    printf("Thrust Throughput: %.3f GEls/s\n", thrustThroughput);
    printf("Speedup: %.2fx\n", thrustThroughput > 0 ? throughput / thrustThroughput : 0);
    
    free(h_Input);
    free(h_Output);
    cudaFree(d_Input);
    cudaFree(d_Output);
    cudaFree(d_HH);
    cudaFree(d_Hg);
    cudaFree(d_SHg);
    cudaFree(d_PSv);
    cudaFree(d_ThrustInput);
    
    return 0;
}
