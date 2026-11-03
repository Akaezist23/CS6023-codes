#!/bin/bash
set -e

# Usage: bash compare_sizes.sh [depth] [threshold]
#   Times the baseline (BASE, default multi_reflections_timed.cu) against an optimized
#   version (OPT, default multi_reflections_shared.cu) for larger scenes, RUNS times each,
#   and prints mean kernel time, speedup, and the marginal cost per extra sphere.
#   Before timing, it renders N=400 with both and checks the images are pixel-identical.
#   Pick the version with OPT, e.g.
#     OPT=multi_reflections_split.cu bash compare_sizes.sh
#   Output is saved to results/compare_sizes_<name>.txt (<name> = OPT without the
#   multi_reflections_ prefix), or to OUT if given. Override the N list with N_VALUES.
#   BLOCK="8 8" passes a block size (arguments 4-5) to both programs; versions that do not
#   read it (e.g. the baseline) ignore it. Compare two optimized versions with BASE, e.g.
#     BASE=multi_reflections_split.cu OPT=multi_reflections_prefetch.cu BLOCK="8 8" bash compare_sizes.sh
#
#   Shared: at most 64 KB per block on the T4 = 1,820 Spheres (36 B each) = floor + 1,819,
#   so it is expected to fail from N = 1820 on (shown as "fail").
#   Split: center + radius as float4 (16 B), so L1 (~64 KB) holds ~4,096 of them;
#   the extra N values 3300-3800 are there to locate its cliff.
BASE=${BASE:-multi_reflections_timed.cu}
OPT=${OPT:-multi_reflections_shared.cu}
[ -f "$BASE" ] || BASE=src/$BASE   # sources live in src/; bare file names also work
[ -f "$OPT" ] || OPT=src/$OPT
NAME=$(basename "$OPT" .cu); NAME=${NAME#multi_reflections_}
DEPTH=${1:-10}
THRESHOLD=${2:-0.01}
N_VALUES=${N_VALUES:-"400 600 800 1000 1200 1364 1365 1500 1700 1819 1820 2000 2500 3000 3300 3600 3800 4000 5000"}
RUNS=3
BLOCK=${BLOCK:-}
OUT=${OUT:-results/compare_sizes_$NAME${BLOCK:+_${BLOCK/ /x}}.txt}
mkdir -p results

nvcc "$BASE" -o mr_base -O2 -std=c++11
nvcc "$OPT"  -o mr_opt  -O2 -std=c++11

# Mean kernel time (ms) over RUNS runs, or "fail" if the program exits with an error.
mean_time() {
    local bin=$1 n=$2 times="" t
    for ((r = 0; r < RUNS; r++)); do
        if ! t=$(./"$bin" "$n" "$DEPTH" "$THRESHOLD" $BLOCK 2>/dev/null | awk '/Kernel time:/ {print $3}') || [ -z "$t" ]; then
            echo fail; return
        fi
        times="$times $t"
    done
    echo $times | awk '{ s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF }'
}

# Correctness: same image at N=400 (each run writes image.ppm)
./mr_base 400 "$DEPTH" "$THRESHOLD" $BLOCK > /dev/null && mv image.ppm image_base.ppm
./mr_opt  400 "$DEPTH" "$THRESHOLD" $BLOCK > /dev/null && mv image.ppm image_opt.ppm
{
echo -n "Image check (N=400): "
python3 compare_images.py image_base.ppm image_opt.ppm || echo "WARNING: $NAME image differs from baseline"

# 128-bit global loads in each binary (the split version should have some in the sphere loop)
echo "LDG.E.128 instructions: baseline $(cuobjdump -sass mr_base | grep -c 'LDG.E.128' || true)," \
     "$NAME $(cuobjdump -sass mr_opt | grep -c 'LDG.E.128' || true)"
} | tee "$OUT"
rm image_base.ppm image_opt.ppm

# Untimed warm-up (GPU wake-up / clock ramp)
./mr_base 10 "$DEPTH" "$THRESHOLD" $BLOCK > /dev/null
./mr_opt  10 "$DEPTH" "$THRESHOLD" $BLOCK > /dev/null

{
echo "baseline = $BASE, $NAME = $OPT${BLOCK:+, block ${BLOCK/ /x}}"
echo "depth $DEPTH, threshold $THRESHOLD, $RUNS runs per N, times in ms"
echo "ms/sph = (time - previous time) / (N - previous N), i.e. cost of each extra sphere (only for steps >= 50)"
printf "%6s %10s %10s %9s %14s %14s\n" "N" "baseline" "$NAME" "speedup" "base ms/sph" "$NAME ms/sph"
PREV_N=""; PREV_B=""; PREV_O=""
for N in $N_VALUES; do
    B=$(mean_time mr_base "$N")
    O=$(mean_time mr_opt "$N")
    awk -v n="$N" -v b="$B" -v o="$O" -v pn="$PREV_N" -v pb="$PREV_B" -v po="$PREV_O" 'BEGIN {
        sp  = (b != "fail" && o != "fail") ? sprintf("%.2fx", b / o) : "-"
        wide = (pn != "" && n - pn >= 50)   # 1-sphere steps (1364->1365, 1819->1820) would just amplify noise
        bms = (wide && b != "fail" && pb != "fail") ? sprintf("%.5f", (b - pb) / (n - pn)) : "-"
        oms = (wide && o != "fail" && po != "fail") ? sprintf("%.5f", (o - po) / (n - pn)) : "-"
        printf "%6d %10s %10s %9s %14s %14s\n", n, b, o, sp, bms, oms
    }'
    PREV_N=$N; PREV_B=$B; PREV_O=$O
done
} | tee -a "$OUT"

rm mr_base mr_opt
echo "Saved to $OUT"
