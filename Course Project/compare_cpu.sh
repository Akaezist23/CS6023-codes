#!/bin/bash
set -e

# Usage: bash compare_cpu.sh [W] [H]
#   CPU vs CUDA on the starter scene (3 spheres + floor, diffuse + hard shadows, no reflections):
#   starter_raytracer_cpu.cpp (g++, one thread) against src/baseline.cu (its direct CUDA port).
#   First checks that the two images match, then times RUNS runs of each:
#     CPU: wall time of the whole process (the starter has no timer; includes writing the PPM,
#          ~1.4 MB, a few ms). The starter is always 800x600; W and H only apply to the GPU.
#     GPU: kernel time (CUDA events) and wall time of the whole process (includes CUDA context
#          setup and copies, which dominate for a scene this small).
#   Change the CPU compiler flags with CXXFLAGS (default -O2, as in run.sh).
#   Output is also saved to results/compare_cpu.txt.
W=${1:-800}
H=${2:-600}
CXXFLAGS=${CXXFLAGS:--O2}
RUNS=${RUNS:-5}   # runs per data point; override with RUNS=20
OUT=${OUT:-results/compare_cpu.txt}
mkdir -p results

g++ $CXXFLAGS -o raytracer_cpu starter_raytracer_cpu.cpp
nvcc src/baseline.cu -o baseline_gpu -O2 -std=c++11

# Wall time of one command in ms
wall_ms() {
    local t0 t1
    t0=$(date +%s%N); "$@" > /dev/null; t1=$(date +%s%N)
    echo $(( (t1 - t0) / 1000 )) | awk '{ printf "%.3f", $1 / 1000 }'
}
# "mean min max" of a list of numbers
stats() {
    echo "$@" | awk '{ s = 0; mn = $1; mx = $1
        for (i = 1; i <= NF; i++) { s += $i; if ($i < mn) mn = $i; if ($i > mx) mx = $i }
        printf "%10.3f %10.3f %10.3f", s / NF, mn, mx }'
}

# Correctness: same image (GPU at 800x600 to match the CPU)
./raytracer_cpu > /dev/null && mv image.ppm image_cpu.ppm
./baseline_gpu 800 600 > /dev/null && mv image.ppm image_gpu.ppm
{
echo -n "Image check CPU vs GPU (800x600): "
python3 compare_images.py image_cpu.ppm image_gpu.ppm || echo "(small differences are expected: the GPU fuses a*b+c into FMA, the CPU build may not)"
} | tee "$OUT"
rm image_cpu.ppm image_gpu.ppm

# Untimed warm-up (GPU wake-up / clock ramp)
./baseline_gpu "$W" "$H" > /dev/null

CPU=""; GPU_K=""; GPU_W=""
for ((r = 0; r < RUNS; r++)); do
    CPU="$CPU $(wall_ms ./raytracer_cpu)"
    GPU_K="$GPU_K $(./baseline_gpu "$W" "$H" | awk '/Kernel time:/ {print $3}')"
    GPU_W="$GPU_W $(wall_ms ./baseline_gpu "$W" "$H")"
done
CPU_MEAN=$(stats $CPU | awk '{print $1}')
K_MEAN=$(stats $GPU_K | awk '{print $1}')
W_MEAN=$(stats $GPU_W | awk '{print $1}')

{
echo "CPU: starter_raytracer_cpu.cpp, g++ $CXXFLAGS, 1 thread, 800x600"
echo "GPU: src/baseline.cu, block 32x32, ${W}x${H}"
echo "$RUNS runs each, times in ms"
printf "%-28s %10s %10s %10s\n" "" "mean" "min" "max"
printf "%-28s %s\n" "CPU (process wall time)" "$(stats $CPU)"
printf "%-28s %s\n" "GPU kernel (CUDA events)" "$(stats $GPU_K)"
printf "%-28s %s\n" "GPU (process wall time)" "$(stats $GPU_W)"
awk -v c="$CPU_MEAN" -v k="$K_MEAN" -v w="$W_MEAN" 'BEGIN {
    printf "speedup CPU / GPU kernel: %.1fx, CPU / GPU process: %.2fx\n", c / k, c / w }'
} | tee -a "$OUT"

rm raytracer_cpu baseline_gpu
echo "Saved to $OUT"
