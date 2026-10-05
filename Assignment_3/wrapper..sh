#!/bin/bash

echo "=========================================="
echo "       Building CUDA program"
echo "=========================================="

nvcc CS23B078.cu -o main

if [ $? -ne 0 ]; then
    echo "Build failed. Tests not run."
    exit 1
fi

echo
echo "Build successful."
echo

./run_tests.s