#!/bin/bash
set -e

# Usage: bash verify_cleanup.sh
#   Checks that the comment cleanup in src/ changed nothing: every .cu file is built twice, from the
#   working tree and from git revision REF (default HEAD, i.e. before the cleanup if it is not
#   committed yet; use REF=HEAD~1 if it is). For each version it compares
#     - the machine code (cuobjdump -sass), which should be identical,
#     - the rendered image (N=5000, depth 10, threshold 0.01, 8x8 blocks where supported),
#     - the mean kernel time over RUNS runs (default 5), old and new interleaved.
REF=${REF:-HEAD}
RUNS=${RUNS:-5}
TMP=verify_tmp
mkdir -p "$TMP"

# Arguments per program (shared fits at most 1,819 spheres; naive/threshold take only the depth)
args_for() {
    case $1 in
        baseline) echo "800 600" ;;
        multi_reflections_naive|multi_reflections_threshold) echo "10" ;;
        multi_reflections_shared) echo "1819 10 0.01" ;;
        multi_reflections_bvh_reps) echo "5000 10 0.01 8 8 800 600 5" ;;
        *) echo "5000 10 0.01 8 8" ;;
    esac
}

kernel_ms() { awk '/Kernel time:/ {print $3}'; }

printf "%-30s %6s %-10s %10s %10s\n" "version" "SASS" "image" "old ms" "new ms"
for SRC in src/*.cu; do
    NAME=$(basename "$SRC" .cu)
    git show "$REF:./$SRC" > "$TMP/$NAME.cu"
    nvcc "$SRC"         -o "$TMP/new" -O2 -std=c++11 -Isrc
    nvcc "$TMP/$NAME.cu" -o "$TMP/old" -O2 -std=c++11 -Isrc
    cuobjdump -sass "$TMP/new" > "$TMP/new.sass"
    cuobjdump -sass "$TMP/old" > "$TMP/old.sass"
    SASS=$(cmp -s "$TMP/new.sass" "$TMP/old.sass" && echo same || echo DIFF)

    ARGS=$(args_for "$NAME")
    "$TMP/old" $ARGS > /dev/null && mv image.ppm "$TMP/old.ppm"
    "$TMP/new" $ARGS > /dev/null && mv image.ppm "$TMP/new.ppm"
    IMG=$(python3 compare_images.py "$TMP/old.ppm" "$TMP/new.ppm" | awk '{print $1}' || true)

    OLD=""; NEW=""
    for ((r = 0; r < RUNS; r++)); do
        OLD="$OLD $("$TMP/old" $ARGS | kernel_ms)"
        NEW="$NEW $("$TMP/new" $ARGS | kernel_ms)"
    done
    mean() { echo $1 | awk '{ if (NF == 0) { print "-"; exit } s = 0; for (i = 1; i <= NF; i++) s += $i; printf "%.3f", s / NF }'; }
    printf "%-30s %6s %-10s %10s %10s\n" "$NAME" "$SASS" "$IMG" "$(mean "$OLD")" "$(mean "$NEW")"
done

rm -rf "$TMP" image.ppm
