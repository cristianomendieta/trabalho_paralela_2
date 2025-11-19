#!/bin/bash
echo "Compiling for 256 threads..."
nvcc -DTHREADS_PER_BLOCK=256U -o segmented-sort-bitonic-256 segmented-sort-bitonic.cu -std=c++14

echo "Compiling for 512 threads..."
nvcc -DTHREADS_PER_BLOCK=512U -o segmented-sort-bitonic-512 segmented-sort-bitonic.cu -std=c++14

echo "Compiling for 1024 threads..."
nvcc -DTHREADS_PER_BLOCK=1024U -o segmented-sort-bitonic-1024 segmented-sort-bitonic.cu -std=c++14

echo "Done."
