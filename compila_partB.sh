#!/bin/bash
echo "Compiling mppSort..."
nvcc -o mppSort mppSort.cu -std=c++14
echo "Done."
