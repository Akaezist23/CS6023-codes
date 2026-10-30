#!/bin/bash
set -e

# Usage: bash profile.sh [N] [depth] [threshold]
#   Profiles ONE render (defaults: N = all 5000, depth 10, threshold 0.01) with nsys and ncu.
#   Always pass N for the shared-memory version: shared memory fits at most ~1,820 spheres.
#   Reports go to profiles/: open the .nsys-rep / .ncu-rep files in the Nsight GUIs,
#   and the .txt files hold the same text that is printed here.
# Pick the source file with SRC and the report folder with OUT, e.g.
#   SRC=multi_reflections_shared.cu OUT=profiles_shared bash profile.sh 400
SRC=${SRC:-multi_reflections_timed.cu}
ARGS="$@"
OUT=${OUT:-profiles}
mkdir -p "$OUT"

# Build with -lineinfo so ncu can map stalls / memory traffic back to source lines.
nvcc "$SRC" -o multi_reflections -O2 -std=c++11 -lineinfo

# ---- 1. nsys: timeline of the whole run (API calls, memcpys, kernel) ----------
echo "========== nsys =========="
nsys profile --trace=cuda --stats=true --force-overwrite=true \
     -o "$OUT/nsys_report" ./multi_reflections $ARGS | tee "$OUT/nsys_stats.txt"

# ---- 2. ncu: full profile of the kernel --------------------------------------
# Note: ncu locks the GPU to its base clock by default, so its kernel durations
# are longer than the ones run.sh measures. Compare ratios, not absolute times.
echo "========== ncu (full set) =========="
ncu --set full --import-source yes --force-overwrite \
    -o "$OUT/ncu_report" ./multi_reflections $ARGS

ncu --import "$OUT/ncu_report.ncu-rep" --page details \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats | tee "$OUT/ncu_details.txt"

# ---- 3. ncu: raw counters for a hand-made roofline ---------------------------
#   FLOPs = fadd + fmul + 2 * ffma   (and the same for doubles, to check none sneak in)
#   bytes = DRAM read + write (also L2 / L1 traffic, for the hierarchical roofline)
echo "========== ncu (roofline counters) =========="
ncu --metrics \
gpu__time_duration.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum \
    ./multi_reflections $ARGS | tee "$OUT/ncu_roofline.txt"

rm multi_reflections
echo "Reports in $OUT/"
