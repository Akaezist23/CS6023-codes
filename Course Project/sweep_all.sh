#!/bin/bash
set -e

# Usage: bash sweep_all.sh [depth] [threshold]
#   Runs compare_all.sh for several scene sizes, so every version is timed (interleaved, RUNS runs)
#   at each N, then merges the per-N tables into one CSV (N, one column per version) for plotting.
#   SCENES=... scene headers in src/ (default: the main scene and kaushik_scene.h),
#   N_VALUES=... the sizes, VERSIONS=... as in compare_all.sh (default: all flat versions except
#   shared, which fails above 1,819 spheres), RUNS=... (default 10).
#   Output per scene: results/sweep_all[_scene]/compare_all_N<N>.{txt,csv} and
#   results/sweep_all[_scene].csv (no suffix for the main scene, scene.h)
SCENES=${SCENES:-"scene.h kaushik_scene.h"}
N_VALUES=${N_VALUES:-"400 1000 1500 2000 2500 3000 3500 4000 4500 5000"}
export VERSIONS=${VERSIONS:-"timed:32x32 split:32x32 split:8x8 2sphere:8x8 prefetch:8x8 fma:8x8"}
export RUNS=${RUNS:-10}

for S in $SCENES; do
    [ "$S" = scene.h ] && S=""   # empty SCENE = the default scene.h, as in compare_all.sh
    export SCENE=$S
    SUFFIX=${SCENE:+_$(basename "$SCENE" .h)}
    DIR=results/sweep_all$SUFFIX
    mkdir -p "$DIR"
    for N in $N_VALUES; do
        echo "=== scene ${SCENE:-scene.h}, N = $N ==="
        N=$N OUT=$DIR/compare_all_N$N.txt bash compare_all.sh "$@"
    done
    python3 merge_sweep.py "$DIR" "$DIR.csv"
done
