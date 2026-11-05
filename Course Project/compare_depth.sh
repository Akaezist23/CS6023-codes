#!/bin/bash
set -e

# Usage: bash compare_depth.sh [threshold]
#   Kernel time as a function of reflection depth, for several versions, RUNS times each.
#   depth d = up to d reflection bounces after the primary hit (0 = no reflections).
#   VERSIONS lists "source:BXxBY" pairs (block defaults to 32x32), e.g. the default
#     VERSIONS="multi_reflections_timed.cu:32x32 multi_reflections_fma.cu:8x8"
#   compares the original baseline with the current best version at its best block size.
#   Each version must take ./prog N depth threshold BX BY.
#   N=... sets the scene size (default 400), DEPTHS=... the depth list.
#   threshold (default 0.01) stops a ray once its energy is below it; with 0, every ray that keeps
#   hitting reflective surfaces runs to the full depth, which shows the pure cost of depth.
#   SAVE_IMAGES=1 also saves the last version's image at every depth to results/depth_images/.
#   Output is saved to results/compare_depth_N<N>_t<threshold>.txt, or to OUT if given.
VERSIONS=${VERSIONS:-"multi_reflections_timed.cu:32x32 multi_reflections_fma.cu:8x8"}
N=${N:-400}
DEPTHS=${DEPTHS:-"0 1 2 3 4 5 6 7 8 9 10"}
THRESHOLD=${1:-0.01}
SAVE_IMAGES=${SAVE_IMAGES:-0}
RUNS=${RUNS:-3}   # runs per data point; override with RUNS=20
SCENE=${SCENE:-}   # e.g. SCENE=scene_adversarial.h (a header in src/); empty = scene.h
NVFLAGS=${SCENE:+-DSCENE_FILE=\"$SCENE\"}
SUFFIX=${SCENE:+_$(basename "$SCENE" .h)}
OUT=${OUT:-results/compare_depth_N${N}_t${THRESHOLD}$SUFFIX.txt}
mkdir -p results

# Build each version as mr_v<i>; LABELS / BLOCKS hold its column name and block size
LABELS=(); BLOCKS=(); i=0
for V in $VERSIONS; do
    SRC=${V%%:*}; BLOCK=32x32; [[ "$V" == *:* ]] && BLOCK=${V#*:}
    [ -f "$SRC" ] || SRC=src/$SRC   # sources live in src/; a bare file name also works
    NAME=$(basename "$SRC" .cu); NAME=${NAME#multi_reflections_}
    nvcc "$SRC" -o mr_v$i -O2 -std=c++11 $NVFLAGS
    LABELS+=("$NAME@$BLOCK"); BLOCKS+=("${BLOCK/x/ }")
    ./mr_v$i 10 10 "$THRESHOLD" ${BLOCKS[$i]} > /dev/null   # untimed warm-up
    i=$((i + 1))
done
LAST=$((i - 1))

# Mean kernel time (ms) over RUNS runs, or "fail" if the program exits with an error.
mean_time() {
    local v=$1 d=$2 times="" t
    for ((r = 0; r < RUNS; r++)); do
        if ! t=$(./mr_v$v "$N" "$d" "$THRESHOLD" ${BLOCKS[$v]} 2>/dev/null | awk '/Kernel time:/ {print $3}') || [ -z "$t" ]; then
            echo fail; return
        fi
        times="$times $t"
    done
    echo $times | awk '{ s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF }'
}

[ "$SAVE_IMAGES" = 1 ] && mkdir -p results/depth_images

{
echo "N $N, threshold $THRESHOLD, $RUNS runs per point, times in ms"
echo "speedup = time(${LABELS[0]}) / time(${LABELS[$LAST]}); step = time(d) - time(d-1) of the last version"
printf "%6s" "depth"; for L in "${LABELS[@]}"; do printf " %18s" "$L"; done; printf " %9s %9s\n" "speedup" "step"
PREV=""
for D in $DEPTHS; do
    printf "%6d" "$D"; TIMES=""
    for ((v = 0; v <= LAST; v++)); do
        T=$(mean_time $v "$D"); TIMES="$TIMES $T"
        printf " %18s" "$T"
    done
    echo $TIMES | awk -v prev="$PREV" '{
        sp = ($1 != "fail" && $NF != "fail") ? sprintf("%.2fx", $1 / $NF) : "-"
        st = (prev != "" && prev != "fail" && $NF != "fail") ? sprintf("%.3f", $NF - prev) : "-"
        printf " %9s %9s\n", sp, st }'
    PREV=$(echo $TIMES | awk '{print $NF}')
    if [ "$SAVE_IMAGES" = 1 ]; then
        ./mr_v$LAST "$N" "$D" "$THRESHOLD" ${BLOCKS[$LAST]} > /dev/null
        python3 -c "from PIL import Image; Image.open('image.ppm').save('results/depth_images/depth_$D.png')"
    fi
done
} | tee "$OUT"

for ((v = 0; v <= LAST; v++)); do rm mr_v$v; done
[ "$SAVE_IMAGES" = 1 ] && echo "Images in results/depth_images/"
python3 table_to_csv.py depth "$OUT"   # same table as CSV, for plotting
echo "Saved to $OUT"
