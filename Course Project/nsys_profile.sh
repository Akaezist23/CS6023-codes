#!/bin/bash
set -e

# Usage: bash nsys_profile.sh [N] [depth] [threshold]
#   Runs ONLY Nsight Systems on one render (no ncu), after an untimed warm-up run,
#   so the kernel time is not inflated by the GPU waking up (profile.sh has no warm-up).
#   Pick the source file with SRC and the report folder with OUT, e.g.
#     SRC=multi_reflections_shared.cu OUT=profiles_shared_1819 bash nsys_profile.sh 1819
#   Writes $OUT/nsys_report.nsys-rep and $OUT/nsys_stats.txt.
SRC=${SRC:-multi_reflections_timed.cu}
OUT=${OUT:-profiles}
ARGS="$@"
mkdir -p "$OUT"

nvcc "$SRC" -o multi_reflections -O2 -std=c++11 -lineinfo

# Untimed warm-up (GPU wake-up / clock ramp)
./multi_reflections 10 > /dev/null

nsys profile --trace=cuda --stats=true --force-overwrite=true \
     -o "$OUT/nsys_report" ./multi_reflections $ARGS | tee "$OUT/nsys_stats.txt"

rm multi_reflections
echo "Reports in $OUT/"
