#!/bin/bash
set -e

# Usage: bash benchmark_final.sh
#   Runs every timing measurement the report uses, with the method of report Sec. 2:
#   kernel time with CUDA events after an in-process warm-up launch, mean of RUNS runs
#   (default 20), every comparison inside one script call (same session).
#   Flat versions only (no BVH). Results go to results/final/<scene>/.
#
#   STAGES selects what to run (default: all, in this order; ~1.5-2 h in total on a T4):
#     cpu         CPU vs GPU on the starter scene (report Sec. 3.3)          ~2 min
#     depth       time vs reflection depth, N=400 (t=0.01, t=0), N=5000      ~10 min  (main scene)
#     resolution  time vs image size, N=400                                  ~5 min   (main scene)
#     all         every flat version interleaved, N=400 and N=5000           ~5 min   per scene
#     sizes       N sweeps: shared/split vs baseline, fma vs split          ~20 min per scene
#     blocks      block-shape sweep of the split version                     ~5 min   per scene
#   SCENES selects the scenes for the per-scene stages (default: all three; depth and
#   resolution always use the main scene only).
#   FP16=1 adds the fp16 version (left out for now): the fp16-vs-fma sweep, and fp16 in "all"
#   (add it to FLAT by hand or via VERSIONS in compare_all.sh).
#   Example: STAGES="all sizes" SCENES="kaushik_scene.h" bash benchmark_final.sh
STAGES=${STAGES:-"cpu depth resolution all sizes blocks"}
SCENES=${SCENES:-"scene.h scene_adversarial.h kaushik_scene.h"}
export RUNS=${RUNS:-20}   # runs per data point
DEPTH=10
THRESHOLD=0.01
FLAT="timed:32x32 split:32x32 split:8x8 2sphere:8x8 prefetch:8x8 fma:8x8"   # fp16: later, see FP16=1
SMALL_N="10 50 100 200 400 600 800 1000 1200 1364 1500 1700 1819 1820"   # shared fits up to 1,819
LARGE_N="400 1000 1500 1819 2000 2500 3000 3300 3600 3800 4000 4500 5000"
MAIN=results/final/scene
has() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo; echo "===== $(date +%H:%M:%S) $* ====="; }

mkdir -p "$MAIN"
if has cpu; then
    log "CPU vs GPU"
    OUT=$MAIN/compare_cpu.txt bash compare_cpu.sh
fi
if has depth; then
    for t in 0.01 0; do
        log "depth, N=400, threshold $t"
        N=400 OUT=$MAIN/compare_depth_N400_t$t.txt bash compare_depth.sh $t
    done
    log "depth, N=5000, threshold 0.01"
    N=5000 OUT=$MAIN/compare_depth_N5000_t0.01.txt bash compare_depth.sh 0.01
fi
if has resolution; then
    log "resolution, N=400"
    N=400 OUT=$MAIN/compare_resolution_N400.txt bash compare_resolution.sh $DEPTH $THRESHOLD
fi

for S in $SCENES; do
    D=results/final/$(basename "$S" .h)
    mkdir -p "$D"
    if has all; then
        log "$S: all flat versions, N=400 (incl. shared)"
        SCENE=$S N=400 VERSIONS="timed:32x32 shared:32x32 $(echo $FLAT | cut -d' ' -f2-)" \
            OUT=$D/compare_all_N400.txt bash compare_all.sh $DEPTH $THRESHOLD
        log "$S: all flat versions, N=5000"
        SCENE=$S N=5000 VERSIONS="$FLAT" OUT=$D/compare_all_N5000.txt bash compare_all.sh $DEPTH $THRESHOLD
    fi
    if has sizes; then
        log "$S: shared vs baseline, 32x32"
        SCENE=$S N_VALUES="$SMALL_N" BASE=multi_reflections_timed.cu OPT=multi_reflections_shared.cu \
            OUT=$D/sizes_shared_vs_timed.txt bash compare_sizes.sh $DEPTH $THRESHOLD
        log "$S: split vs baseline, 32x32"
        SCENE=$S N_VALUES="$LARGE_N" BASE=multi_reflections_timed.cu OPT=multi_reflections_split.cu \
            OUT=$D/sizes_split_vs_timed.txt bash compare_sizes.sh $DEPTH $THRESHOLD
        log "$S: fma vs split, 8x8"
        SCENE=$S N_VALUES="$LARGE_N" BASE=multi_reflections_split.cu OPT=multi_reflections_fma.cu BLOCK="8 8" \
            OUT=$D/sizes_fma_vs_split_8x8.txt bash compare_sizes.sh $DEPTH $THRESHOLD
        if [ "${FP16:-0}" = 1 ]; then
        log "$S: fp16 vs fma, 8x8"
        SCENE=$S N_VALUES="$LARGE_N" BASE=multi_reflections_fma.cu OPT=multi_reflections_fp16.cu BLOCK="8 8" \
            OUT=$D/sizes_fp16_vs_fma_8x8.txt bash compare_sizes.sh $DEPTH $THRESHOLD
        fi
    fi
    if has blocks; then
        log "$S: block shapes, split"
        SCENE=$S SRC=multi_reflections_split.cu OUT=$D/blocks_split.txt bash compare_blocks.sh $DEPTH $THRESHOLD
    fi
done

log "done"
echo "Results in results/final/. Download with:"
echo "  zip -qr results_final.zip results/final"
