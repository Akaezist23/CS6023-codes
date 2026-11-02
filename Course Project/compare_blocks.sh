#!/bin/bash
set -e

# Usage: bash compare_blocks.sh [depth] [threshold]
#   Times one version (SRC, default multi_reflections_split.cu) with different block shapes,
#   RUNS times each, for a few scene sizes, and prints the speedup over the 32x32 default.
#   SRC must take the block size as arguments 4 and 5: ./prog N depth threshold BX BY.
#   Every shape is first checked to render the same image as 32x32 (N=400).
#   Output is saved to results/compare_blocks_<name>.txt, or to OUT if given.
#   Override the lists with N_VALUES and SHAPES (BXxBY, the first one is the reference).
#
#   32x8 keeps the current warp shape (one 32-pixel row) and only changes block granularity;
#   16x16 also changes the warp shape to 16x2 pixels (may change divergence).
SRC=${SRC:-multi_reflections_split.cu}
NAME=$(basename "$SRC" .cu); NAME=${NAME#multi_reflections_}
DEPTH=${1:-10}
THRESHOLD=${2:-0.01}
N_VALUES=${N_VALUES:-"400 3000 5000"}
SHAPES=${SHAPES:-"32x32 32x16 32x8 32x4 16x16 16x8 8x8"}
RUNS=3
OUT=${OUT:-results/compare_blocks_$NAME.txt}
mkdir -p results

nvcc "$SRC" -o mr_blocks -O2 -std=c++11

# Mean kernel time (ms) over RUNS runs, or "fail" if the program exits with an error.
mean_time() {
    local n=$1 bx=$2 by=$3 times="" t
    for ((r = 0; r < RUNS; r++)); do
        if ! t=$(./mr_blocks "$n" "$DEPTH" "$THRESHOLD" "$bx" "$by" 2>/dev/null | awk '/Kernel time:/ {print $3}') || [ -z "$t" ]; then
            echo fail; return
        fi
        times="$times $t"
    done
    echo $times | awk '{ s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF }'
}

REF=${SHAPES%% *}

# Correctness: every shape renders the same image as the reference shape (N=400)
./mr_blocks 400 "$DEPTH" "$THRESHOLD" ${REF/x/ } > /dev/null && mv image.ppm image_ref.ppm
for S in $SHAPES; do
    [ "$S" = "$REF" ] && continue
    ./mr_blocks 400 "$DEPTH" "$THRESHOLD" ${S/x/ } > /dev/null
    echo -n "Image check $S vs $REF (N=400): "
    python3 compare_images.py image_ref.ppm image.ppm || echo "WARNING: $S image differs"
done
rm -f image_ref.ppm

# Untimed warm-up (GPU wake-up / clock ramp)
./mr_blocks 10 "$DEPTH" "$THRESHOLD" > /dev/null

{
echo "$NAME = $SRC, depth $DEPTH, threshold $THRESHOLD, $RUNS runs per point, times in ms"
echo "speedup = time($REF) / time(shape) at the same N"
printf "%6s" "N"; for S in $SHAPES; do printf " %10s" "$S"; done; echo
for N in $N_VALUES; do
    printf "%6d" "$N"; TIMES=""
    for S in $SHAPES; do
        T=$(mean_time "$N" ${S/x/ }); TIMES="$TIMES $T"
        printf " %10s" "$T"
    done
    echo
    printf "%6s" ""
    echo $TIMES | awk '{ for (i = 1; i <= NF; i++)
        printf " %10s", ($1 != "fail" && $i != "fail") ? sprintf("%.2fx", $1 / $i) : "-"; print "" }'
done
} | tee "$OUT"

rm mr_blocks
echo "Saved to $OUT"
