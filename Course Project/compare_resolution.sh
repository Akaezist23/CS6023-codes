#!/bin/bash
set -e

# Usage: bash compare_resolution.sh [depth] [threshold]
#   Kernel time as a function of image resolution, for several versions, RUNS times each.
#   VERSIONS lists "source:BXxBY" pairs (block defaults to 32x32), e.g. the default
#     VERSIONS="multi_reflections_timed.cu:32x32 multi_reflections_fma.cu:8x8"
#   Each version must take ./prog N depth threshold BX BY W H.
#   All default resolutions are 4:3, so the view (and the work per pixel) stays the same and
#   the time should grow with the pixel count; the "/Mpix" rows show the cost per megapixel.
#   N=... sets the scene size (default 400), RESOLUTIONS=... the WxH list.
#   Output is saved to results/compare_resolution_N<N>.txt, or to OUT if given.
VERSIONS=${VERSIONS:-"multi_reflections_timed.cu:32x32 multi_reflections_fma.cu:8x8"}
N=${N:-400}
RESOLUTIONS=${RESOLUTIONS:-"320x240 640x480 800x600 1280x960 1600x1200 2400x1800 3200x2400"}
DEPTH=${1:-10}
THRESHOLD=${2:-0.01}
RUNS=${RUNS:-3}   # runs per data point; override with RUNS=20
SCENE=${SCENE:-}   # e.g. SCENE=scene_adversarial.h (a header in src/); empty = scene.h
NVFLAGS=${SCENE:+-DSCENE_FILE=\"$SCENE\"}
SUFFIX=${SCENE:+_$(basename "$SCENE" .h)}
OUT=${OUT:-results/compare_resolution_N${N}$SUFFIX.txt}
mkdir -p results

# Build each version as mr_v<i>; LABELS / BLOCKS hold its column name and block size
LABELS=(); BLOCKS=(); i=0
for V in $VERSIONS; do
    SRC=${V%%:*}; BLOCK=32x32; [[ "$V" == *:* ]] && BLOCK=${V#*:}
    [ -f "$SRC" ] || SRC=src/$SRC   # sources live in src/; a bare file name also works
    NAME=$(basename "$SRC" .cu); NAME=${NAME#multi_reflections_}
    nvcc "$SRC" -o mr_v$i -O2 -std=c++11 $NVFLAGS
    LABELS+=("$NAME@$BLOCK"); BLOCKS+=("${BLOCK/x/ }")
    ./mr_v$i 10 "$DEPTH" "$THRESHOLD" ${BLOCKS[$i]} > /dev/null   # untimed warm-up
    i=$((i + 1))
done
LAST=$((i - 1))

# Mean kernel time (ms) over RUNS runs, or "fail" if the program exits with an error.
mean_time() {
    local v=$1 w=$2 h=$3 times="" t
    for ((r = 0; r < RUNS; r++)); do
        if ! t=$(./mr_v$v "$N" "$DEPTH" "$THRESHOLD" ${BLOCKS[$v]} "$w" "$h" 2>/dev/null | awk '/Kernel time:/ {print $3}') || [ -z "$t" ]; then
            echo fail; return
        fi
        times="$times $t"
    done
    echo $times | awk '{ s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF }'
}

{
echo "N $N, depth $DEPTH, threshold $THRESHOLD, $RUNS runs per point, times in ms"
echo "speedup = time(${LABELS[0]}) / time(${LABELS[$LAST]}); /Mpix = ms per million pixels"
printf "%10s %6s" "WxH" "Mpix"; for L in "${LABELS[@]}"; do printf " %18s" "$L"; done; printf " %9s\n" "speedup"
for R in $RESOLUTIONS; do
    W=${R%x*}; H=${R#*x}
    MPIX=$(awk -v w="$W" -v h="$H" 'BEGIN { printf "%.3f", w * h / 1e6 }')
    printf "%10s %6s" "$R" "$MPIX"; TIMES=""
    for ((v = 0; v <= LAST; v++)); do
        T=$(mean_time $v "$W" "$H"); TIMES="$TIMES $T"
        printf " %18s" "$T"
    done
    echo $TIMES | awk '{ printf " %9s\n", ($1 != "fail" && $NF != "fail") ? sprintf("%.2fx", $1 / $NF) : "-" }'
    printf "%10s %6s" "" "/Mpix"
    echo $TIMES | awk -v m="$MPIX" '{ for (i = 1; i <= NF; i++)
        printf " %18s", ($i != "fail") ? sprintf("%.3f", $i / m) : "-"; print "" }'
done
} | tee "$OUT"

for ((v = 0; v <= LAST; v++)); do rm mr_v$v; done
python3 table_to_csv.py resolution "$OUT"   # same table as CSV, for plotting
echo "Saved to $OUT"
