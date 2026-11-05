#!/bin/bash
set -e

# Usage: bash compare_all.sh [depth] [threshold]
#   Times every version at one scene size in one session, so all numbers are directly comparable.
#   Runs are interleaved (round r runs every version once), so clock drift during the session
#   affects all versions alike. Each version's image is compared with the first version's.
#   VERSIONS lists "name:BXxBY" pairs; name = multi_reflections_<name>.cu in src/ (or a file name).
#   The first entry is the reference for "vs ref"; "vs prev" compares with the row above.
#   N=... scene size (default 5000), SCENE=... scene header in src/ (default scene.h),
#   RUNS=... runs per version (default 10). Output: results/compare_all_N<N>_t<threshold>[_scene].txt
#   Not included: shared (fails above 1,820 spheres), naive / threshold (no kernel timing),
#   bvh_reps (times 20 in-process launches, a different method).
VERSIONS=${VERSIONS:-"timed:32x32 split:32x32 split:8x8 2sphere:8x8 prefetch:8x8 fma:8x8 bvh_bare:8x8 bvh:8x8"}
N=${N:-5000}
DEPTH=${1:-10}
THRESHOLD=${2:-0.01}
RUNS=${RUNS:-10}   # runs per data point; override with RUNS=20
SCENE=${SCENE:-}   # e.g. SCENE=kaushik_scene.h (a header in src/); empty = scene.h
NVFLAGS=${SCENE:+-DSCENE_FILE=\"$SCENE\"}
SUFFIX=${SCENE:+_$(basename "$SCENE" .h)}
OUT=${OUT:-results/compare_all_N${N}_t${THRESHOLD}$SUFFIX.txt}
mkdir -p results

# Build each version as mr_v<i>; LABELS / BLOCKS hold its name and block size
LABELS=(); BLOCKS=(); i=0
for V in $VERSIONS; do
    SRC=${V%%:*}; BLOCK=32x32; [[ "$V" == *:* ]] && BLOCK=${V#*:}
    [ -f "$SRC" ] || [ -f "src/$SRC" ] || SRC=multi_reflections_$SRC.cu
    [ -f "$SRC" ] || SRC=src/$SRC   # sources live in src/; a bare file name also works
    NAME=$(basename "$SRC" .cu); NAME=${NAME#multi_reflections_}
    nvcc "$SRC" -o mr_v$i -O2 -std=c++11 $NVFLAGS
    LABELS+=("$NAME"); BLOCKS+=("${BLOCK/x/ }")
    i=$((i + 1))
done
LAST=$((i - 1))

# Correctness: every version's image vs the first one's (same N, depth, threshold)
{
echo "N $N, depth $DEPTH, threshold $THRESHOLD, scene ${SCENE:-scene.h}, $RUNS interleaved runs per version"
./mr_v0 "$N" "$DEPTH" "$THRESHOLD" ${BLOCKS[0]} > /dev/null && mv image.ppm image_ref.ppm
for ((v = 1; v <= LAST; v++)); do
    ./mr_v$v "$N" "$DEPTH" "$THRESHOLD" ${BLOCKS[$v]} > /dev/null
    echo -n "Image check ${LABELS[$v]}@${BLOCKS[$v]/ /x} vs ${LABELS[0]}@${BLOCKS[0]/ /x}: "
    python3 compare_images.py image_ref.ppm image.ppm || true
done
rm -f image_ref.ppm image.ppm
} | tee "$OUT"

# Untimed warm-up of every version (GPU wake-up / clock ramp)
for ((v = 0; v <= LAST; v++)); do ./mr_v$v 10 "$DEPTH" "$THRESHOLD" ${BLOCKS[$v]} > /dev/null; done

# Interleaved timing: TIMES[v] collects the kernel times of version v ("fail" if it errors)
TIMES=()
for ((r = 0; r < RUNS; r++)); do
    for ((v = 0; v <= LAST; v++)); do
        t=$(./mr_v$v "$N" "$DEPTH" "$THRESHOLD" ${BLOCKS[$v]} 2>/dev/null | awk '/Kernel time:/ {print $3}') || t=""
        TIMES[$v]="${TIMES[$v]} ${t:-fail}"
    done
done

{
echo "times in ms; vs ref = time(${LABELS[0]}) / time; vs prev = time(row above) / time"
printf "%-12s %6s %10s %8s %10s %10s %9s %9s\n" "version" "block" "mean" "std" "min" "max" "vs ref" "vs prev"
REF=""; PREV=""
for ((v = 0; v <= LAST; v++)); do
    STATS=$(echo ${TIMES[$v]} | awk '/fail/ { print "fail"; exit }
        { min = $1; max = $1; s = 0; q = 0
          for (i = 1; i <= NF; i++) { s += $i; q += $i * $i; if ($i < min) min = $i; if ($i > max) max = $i }
          m = s / NF; var = q / NF - m * m; if (var < 0) var = 0
          printf "%.3f %.3f %.3f %.3f", m, sqrt(var), min, max }')
    MEAN=${STATS%% *}
    [ -z "$REF" ] && REF=$MEAN
    awk -v l="${LABELS[$v]}" -v b="${BLOCKS[$v]/ /x}" -v st="$STATS" -v m="$MEAN" -v r="$REF" -v p="$PREV" 'BEGIN {
        if (st == "fail") { printf "%-12s %6s %10s\n", l, b, "fail"; exit }
        split(st, a, " ")
        vr = (r != "fail") ? sprintf("%.2fx", r / m) : "-"
        vp = (p != "" && p != "fail") ? sprintf("%.2fx", p / m) : "-"
        printf "%-12s %6s %10s %8s %10s %10s %9s %9s\n", l, b, a[1], a[2], a[3], a[4], vr, vp
    }'
    PREV=$MEAN
done
} | tee -a "$OUT"

for ((v = 0; v <= LAST; v++)); do rm mr_v$v; done
echo "Saved to $OUT"
