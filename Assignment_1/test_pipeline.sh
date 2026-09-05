#!/usr/bin/env bash
#
# test_pipeline.sh -- test harness for CS6023 Assignment 1 ("The Wizard's Lens")
#
# Usage:
#   ./test_pipeline.sh <your_file.cu> [num_random_cases]
#
# Example:
#   ./test_pipeline.sh CS24S009.cu 10
#
# What it does:
#   1. Compiles your .cu file with nvcc.
#   2. Runs a set of hand-picked EDGE-CASE tests (1x1 images, upscaling,
#      no resize, odd crop offsets, etc.)
#   3. Runs `num_random_cases` RANDOM tests with random dimensions.
#   4. For each test: generates an input, runs your binary on it, runs
#      the Python reference on it, and compares the two outputs using
#      the assignment's tolerance rule.
#
# Requires: nvcc, python3 (with generate_input.py, reference.py,
#           compare_outputs.py in the same directory as this script).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/generate_input.py"
REF="$SCRIPT_DIR/reference.py"
CMP="$SCRIPT_DIR/compare_outputs.py"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <your_file.cu> [num_random_cases]"
    exit 2
fi

CU_FILE="$1"
NUM_RANDOM="${2:-10}"

if [ ! -f "$CU_FILE" ]; then
    echo "Error: file '$CU_FILE' not found."
    exit 2
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

BIN="$WORKDIR/pipeline_bin"

echo "==> Compiling $CU_FILE ..."
nvcc -O2 -o "$BIN" "$CU_FILE"
if [ $? -ne 0 ]; then
    echo "Compilation FAILED."
    exit 1
fi
echo "==> Compiled successfully."
echo

PASS_COUNT=0
FAIL_COUNT=0

run_case () {
    local name="$1"
    local input_file="$2"

    local actual_out="$WORKDIR/actual_${name}.txt"
    local expected_out="$WORKDIR/expected_${name}.txt"

    "$BIN" "$input_file" > "$actual_out" 2>"$WORKDIR/stderr_${name}.txt"
    if [ $? -ne 0 ]; then
        echo "[$name] FAIL: your binary crashed / exited non-zero. See stderr:"
        sed 's/^/      /' "$WORKDIR/stderr_${name}.txt"
        FAIL_COUNT=$((FAIL_COUNT+1))
        return
    fi

    python3 "$REF" "$input_file" > "$expected_out"

    result="$(python3 "$CMP" "$actual_out" "$expected_out")"
    status=$?
    if [ $status -eq 0 ]; then
        echo "[$name] $result"
        PASS_COUNT=$((PASS_COUNT+1))
    else
        echo "[$name] $result"
        echo "      input:    $input_file"
        echo "      actual:   $actual_out"
        echo "      expected: $expected_out"
        FAIL_COUNT=$((FAIL_COUNT+1))
    fi
}

echo "==> Running edge-case tests ..."

# name  H   W   Hr  Wr  Hc  Wc  mean  std  seed
declare -a EDGE_CASES=(
    "tiny_1x1                1   1   1   1   1   1   0.5  0.5  1"
    "no_resize_no_crop       16  16  16  16  16  16  0.45 0.22 2"
    "upscale_2x              8   8   16  16  16  16  0.5  0.5  3"
    "downscale               64  64  16  16  16  16  0.5  0.25 4"
    "nonsquare_image         30  50  40  40  32  32  0.5  0.5  5"
    "odd_crop_offset         20  20  9   9   4   4   0.5  0.5  6"
    "resize_1row             5   5   1   5   1   5   0.5  0.5  7"
    "resize_1col             5   5   5   1   5   1   0.5  0.5  8"
    "crop_equals_resize      12  12  20  30  20  30  0.5  0.5  9"
    "extreme_mean_std        10  10  10  10  8   8   1.0  0.01 10"
    "max_size_no_resize      1024 1024 1024 1024 1000 1000 0.5 0.5 11"
    "max_size_downscale      1024 1024 224  224  200  200  0.45 0.22 12"
    "max_size_upscale        64   64  1024 1024 900  900  0.5  0.5  13"
    "max_size_extreme_upscale 2   2   1024 1024 1024 1024 0.5  0.5  14"
    "max_size_nonsquare      1024 512 700  900  650  850  0.5  0.5  15"
)

for case_spec in "${EDGE_CASES[@]}"; do
    read -r name H W Hr Wr Hc Wc mean std seed <<< "$case_spec"
    infile="$WORKDIR/in_${name}.txt"
    python3 "$GEN" "$H" "$W" "$Hr" "$Wr" "$Hc" "$Wc" "$mean" "$std" "$seed" > "$infile"
    run_case "$name" "$infile"
done

echo
echo "==> Running $NUM_RANDOM random tests ..."
for i in $(seq 1 "$NUM_RANDOM"); do
    H=$(( (RANDOM % 100) + 1 ))
    W=$(( (RANDOM % 100) + 1 ))
    Hr=$(( (RANDOM % 100) + 1 ))
    Wr=$(( (RANDOM % 100) + 1 ))
    Hc=$(( (RANDOM % Hr) + 1 ))
    Wc=$(( (RANDOM % Wr) + 1 ))
    mean="0.$((RANDOM % 100))"
    std="0.$(( (RANDOM % 199) + 1 ))"
    seed=$((RANDOM))

    name="random_$i"
    infile="$WORKDIR/in_${name}.txt"
    python3 "$GEN" "$H" "$W" "$Hr" "$Wr" "$Hc" "$Wc" "$mean" "$std" "$seed" > "$infile"
    run_case "$name" "$infile"
done

echo
echo "==================================================="
echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "==================================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
