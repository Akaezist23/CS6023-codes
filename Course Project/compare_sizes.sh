#!/bin/bash
set -e

# Usage: bash compare_sizes.sh [depth] [threshold]
#   Times the baseline (multi_reflections_timed.cu) and the shared-memory version
#   (multi_reflections_shared.cu) for larger scenes, RUNS times each, and prints
#   mean kernel time, speedup, and the marginal cost per extra sphere.
#   Shared memory holds at most 64 KB per block on the T4 = 1,820 Spheres (36 B each)
#   = floor + 1,819, so the shared version is expected to fail from N = 1820 on
#   (shown as "fail"). Above that only the baseline runs, to look for the point where
#   the scene stops fitting in L1 (~64 KB, i.e. also around N ~ 1,800).
#   Output is also saved to results/compare_sizes.txt.
DEPTH=${1:-10}
THRESHOLD=${2:-0.01}
N_VALUES="400 600 800 1000 1200 1364 1365 1500 1700 1819 1820 2000 2500 3000 4000 5000"
RUNS=3
OUT=results/compare_sizes.txt
mkdir -p results

nvcc multi_reflections_timed.cu  -o mr_base   -O2 -std=c++11
nvcc multi_reflections_shared.cu -o mr_shared -O2 -std=c++11

# Mean kernel time (ms) over RUNS runs, or "fail" if the program exits with an error.
mean_time() {
    local bin=$1 n=$2 times="" t
    for ((r = 0; r < RUNS; r++)); do
        if ! t=$(./"$bin" "$n" "$DEPTH" "$THRESHOLD" 2>/dev/null | awk '/Kernel time:/ {print $3}') || [ -z "$t" ]; then
            echo fail; return
        fi
        times="$times $t"
    done
    echo $times | awk '{ s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF }'
}

# Untimed warm-up (GPU wake-up / clock ramp)
./mr_base 10 "$DEPTH" "$THRESHOLD" > /dev/null
./mr_shared 10 "$DEPTH" "$THRESHOLD" > /dev/null

{
echo "depth $DEPTH, threshold $THRESHOLD, $RUNS runs per N, times in ms"
echo "ms/sph = (time - previous time) / (N - previous N), i.e. cost of each extra sphere (only for steps >= 50)"
printf "%6s %10s %10s %9s %14s %14s\n" "N" "baseline" "shared" "speedup" "base ms/sph" "shared ms/sph"
PREV_N=""; PREV_B=""; PREV_S=""
for N in $N_VALUES; do
    B=$(mean_time mr_base "$N")
    S=$(mean_time mr_shared "$N")
    awk -v n="$N" -v b="$B" -v s="$S" -v pn="$PREV_N" -v pb="$PREV_B" -v ps="$PREV_S" 'BEGIN {
        sp  = (b != "fail" && s != "fail") ? sprintf("%.2fx", b / s) : "-"
        wide = (pn != "" && n - pn >= 50)   # 1-sphere steps (1364->1365, 1819->1820) would just amplify noise
        bms = (wide && b != "fail" && pb != "fail") ? sprintf("%.5f", (b - pb) / (n - pn)) : "-"
        sms = (wide && s != "fail" && ps != "fail") ? sprintf("%.5f", (s - ps) / (n - pn)) : "-"
        printf "%6d %10s %10s %9s %14s %14s\n", n, b, s, sp, bms, sms
    }'
    PREV_N=$N; PREV_B=$B; PREV_S=$S
done
} | tee "$OUT"

rm mr_base mr_shared
echo "Saved to $OUT"
