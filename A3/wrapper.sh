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

N=1

# Find -n and its value
if [[ "$*" =~ -n[[:space:]]+([0-9]+) ]]; then
    N="${BASH_REMATCH[1]}"
fi

# Remove -n N from the arguments
ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "-n" ]]; then
        shift 2
    else
        ARGS+=("$1")
        shift
    fi
done

# Repeat the entire test sequence N times
for ((i=1; i<=N; i++)); do
    echo "========== Run $i/$N =========="
    ./run_tests.sh "${ARGS[@]}"
done