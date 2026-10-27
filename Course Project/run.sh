#!/bin/bash
set -e

# Usage: bash run.sh [depth] [threshold]
#   Sweeps N (spheres excluding the floor) over N_VALUES, RUNS times each,
#   and prints kernel-time stats (ms) per N.
#   depth: reflection depth (default 10)
#   threshold: energy threshold (default 0.01)
DEPTH=${1:-10}
THRESHOLD=${2:-0.01}
N_VALUES="10 20 50 100 200 300 400"
RUNS=5

# Build
# g++ -O2 -o raytracer_cpu starter_raytracer_cpu.cpp
# nvcc baseline.cu -o baseline -O2 -std=c++11
nvcc multi_reflections_timed.cu -o multi_reflections -O2 -std=c++11

# Run (each run writes image.ppm; the last one, N = 400, is kept)
# ./raytracer_cpu
# ./baseline
./multi_reflections 10 "$DEPTH" "$THRESHOLD" > /dev/null   # untimed warm-up (GPU wake-up / clock ramp)
echo "depth $DEPTH, threshold $THRESHOLD, $RUNS runs per N"
printf "%6s %10s %10s %10s %10s\n" "N" "mean" "std" "min" "max"
for N in $N_VALUES; do
    TIMES=""
    for ((r = 0; r < RUNS; r++)); do
        T=$(./multi_reflections "$N" "$DEPTH" "$THRESHOLD" | awk '/Kernel time:/ {print $3}')
        TIMES="$TIMES $T"
    done
    echo $TIMES | awk -v n="$N" '{
        min = $1; max = $1; sum = 0; sq = 0
        for (i = 1; i <= NF; i++) {
            sum += $i; sq += $i * $i
            if ($i < min) min = $i
            if ($i > max) max = $i
        }
        mean = sum / NF
        var = sq / NF - mean * mean; if (var < 0) var = 0
        printf "%6d %10.3f %10.3f %10.3f %10.3f\n", n, mean, sqrt(var), min, max
    }'
done

# rm raytracer_cpu
# rm baseline
rm multi_reflections

# Convert image.ppm -> image.png (PIL is preinstalled on Colab)
python3 -c "from PIL import Image; Image.open('image.ppm').save('image.png')"
echo "Wrote image.png"

# A shell started with `!bash run.sh` can't draw into the Colab output cell,
# so display image.png from Python in the same cell:
#   !bash run.sh 10
#   from IPython.display import Image, display; display(Image('image.png'))
