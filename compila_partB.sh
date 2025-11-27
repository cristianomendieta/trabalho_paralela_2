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

echo "Compiling mppSort..."
nvcc -o mppSort mppSort.cu $FLAGS
echo "Done."
