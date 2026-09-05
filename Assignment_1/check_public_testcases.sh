#!/usr/bin/env bash
#
# check_public_testcases.sh -- runs input_01.txt..input_05.txt (the
# official public test cases shipped with the assignment) through
# reference.py and compares against the official expected_01.txt..
# expected_05.txt files. This is a sanity check that reference.py
# itself is correct, using ground truth you did NOT generate yourself.
#
# Optionally, if you also pass your compiled/compilable .cu file, it
# will additionally run YOUR binary on the same 5 official inputs and
# compare against the same official expected outputs -- a completely
# independent check of your CUDA code that doesn't rely on reference.py
# at all.
#
# Usage:
#   ./check_public_testcases.sh <public_testcases_dir> [your_file.cu]
#
# Expects <public_testcases_dir> to contain:
#   input_01.txt input_02.txt input_03.txt input_04.txt input_05.txt
#   expected_01.txt expected_02.txt expected_03.txt expected_04.txt expected_05.txt
#
# (matches the naming used in the assignment's testcases/public folder)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REF="$SCRIPT_DIR/reference.py"
CMP="$SCRIPT_DIR/compare_outputs.py"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <public_testcases_dir> [your_file.cu]"
    exit 2
fi

TESTDIR="$1"
CU_FILE="${2:-}"

if [ ! -d "$TESTDIR" ]; then
    echo "Error: directory '$TESTDIR' not found."
    exit 2
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

BIN=""
if [ -n "$CU_FILE" ]; then
    if [ ! -f "$CU_FILE" ]; then
        echo "Error: file '$CU_FILE' not found."
        exit 2
    fi
    BIN="$WORKDIR/pipeline_bin"
    echo "==> Compiling $CU_FILE ..."
    nvcc -O2 -o "$BIN" "$CU_FILE"
    if [ $? -ne 0 ]; then
        echo "Compilation FAILED."
        exit 1
    fi
    echo "==> Compiled successfully."
    echo
fi

REF_PASS=0
REF_FAIL=0
CU_PASS=0
CU_FAIL=0

for i in 01 02 03 04 05; do
    infile="$TESTDIR/input_${i}.txt"
    expfile="$TESTDIR/expected_${i}.txt"

    if [ ! -f "$infile" ]; then
        echo "[$i] SKIP: missing $infile"
        continue
    fi
    if [ ! -f "$expfile" ]; then
        echo "[$i] SKIP: missing $expfile"
        continue
    fi

    # --- Sanity check reference.py itself against the official expected output ---
    ref_out="$WORKDIR/ref_out_${i}.txt"
    python3 "$REF" "$infile" > "$ref_out"
    ref_result="$(python3 "$CMP" "$ref_out" "$expfile")"
    ref_status=$?
    if [ $ref_status -eq 0 ]; then
        REF_PASS=$((REF_PASS+1))
    else
        REF_FAIL=$((REF_FAIL+1))
    fi
    echo "[$i] reference.py vs official expected_${i}.txt: $ref_result"

    # --- Optionally also check your compiled CUDA binary directly ---
    if [ -n "$BIN" ]; then
        cu_out="$WORKDIR/cu_out_${i}.txt"
        "$BIN" "$infile" > "$cu_out" 2>"$WORKDIR/stderr_${i}.txt"
        if [ $? -ne 0 ]; then
            echo "[$i] your binary vs official expected_${i}.txt: FAIL (crashed / non-zero exit)"
            sed 's/^/      /' "$WORKDIR/stderr_${i}.txt"
            CU_FAIL=$((CU_FAIL+1))
        else
            cu_result="$(python3 "$CMP" "$cu_out" "$expfile")"
            cu_status=$?
            echo "[$i] your binary  vs official expected_${i}.txt: $cu_result"
            if [ $cu_status -eq 0 ]; then
                CU_PASS=$((CU_PASS+1))
            else
                CU_FAIL=$((CU_FAIL+1))
            fi
        fi
    fi
    echo
done

echo "==================================================="
echo "  reference.py:  $REF_PASS passed, $REF_FAIL failed  (out of official public tests)"
if [ -n "$BIN" ]; then
    echo "  your binary:   $CU_PASS passed, $CU_FAIL failed  (out of official public tests)"
fi
echo "==================================================="

if [ "$REF_FAIL" -gt 0 ] || [ "$CU_FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
