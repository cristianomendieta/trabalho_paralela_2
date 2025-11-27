#!/bin/bash
set -e

HOST=$(hostname)
echo "Compiling on machine: $HOST"

FLAGS=""
if [ "$HOST" = "nv00" ]; then
    FLAGS="--std=c++14 -I /usr/include/c++/10 -I /usr/lib/cuda/include/"
elif [ "$HOST" = "orval" ]; then
    FLAGS="-arch sm_50 --allow-unsupported-compiler -std=c++17 -Xcompiler=-std=c++17 -ccbin /usr/bin/g++-12"
else
    FLAGS="-std=c++14"
fi

echo "Using flags: $FLAGS"

echo "Compiling for 256 threads..."
nvcc -DTHREADS_PER_BLOCK=256U $FLAGS -o segmented-sort-bitonic-256 segmented-sort-bitonic.cu

echo "Compiling for 512 threads..."
nvcc -DTHREADS_PER_BLOCK=512U $FLAGS -o segmented-sort-bitonic-512 segmented-sort-bitonic.cu

echo "Compiling for 1024 threads..."
nvcc -DTHREADS_PER_BLOCK=1024U $FLAGS -o segmented-sort-bitonic-1024 segmented-sort-bitonic.cu

echo "Done."
