#!/bin/bash

# Configuration
SOLVER_EXEC="./main"
INPUT_DIR="input"
EXPECTED_DIR="output"
TEMP_DIR="temp_output"

# ANSI Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color


# ============================================================
# Check executable
# ============================================================

if [ ! -f "$SOLVER_EXEC" ]; then
    echo -e "${RED}Error: Executable '$SOLVER_EXEC' not found! Build your code first.${NC}"
    exit 1
fi


# ============================================================
# Create temporary output directory
# ============================================================

mkdir -p "$TEMP_DIR"


TOTAL=0
PASSED=0
FAILED=0


echo "=========================================="
echo "    Running CUDA Delta-Stepping Tests     "
echo "=========================================="


# ============================================================
# Run a single test
# ============================================================

run_test() {
    local test_num="$1"

    local input_file="$INPUT_DIR/$test_num"
    local expected_file="$EXPECTED_DIR/$test_num"
    local temp_file="$TEMP_DIR/$test_num"

    # Check input file
    if [ ! -f "$input_file" ]; then
        echo -e "[${YELLOW}SKIP${NC}] Test $test_num -> Input file not found"
        return
    fi

    ((TOTAL++))

    # Check expected output
    if [ ! -f "$expected_file" ]; then
        echo -e "[${YELLOW}SKIP${NC}] Test $test_num -> Missing expected output"
        return
    fi

    # Run solver
    $SOLVER_EXEC "$input_file" "$temp_file" 

    # Compare outputs, ignoring whitespace and blank lines
    if diff -w -B "$temp_file" "$expected_file" > /dev/null 2>&1; then
        echo -e "[${GREEN}PASS${NC}] Test $test_num"
        ((PASSED++))
    else
        echo -e "[${RED}FAIL${NC}] Test $test_num"
        ((FAILED++))
    fi
}


# ============================================================
# Determine which tests to run
#
# No arguments:
#     ./run_tests.sh
#     -> runs every test in input/
#
# Individual tests:
#     ./run_tests.sh 0
#     ./run_tests.sh 0 3 7
#
# Ranges:
#     ./run_tests.sh 0-5
#     ./run_tests.sh 0 3-7 10
# ============================================================

if [ $# -eq 0 ]; then

    # Run every test in input/
    for input_file in "$INPUT_DIR"/*; do
        [ -e "$input_file" ] || continue

        test_num=$(basename "$input_file")

        # Only consider numeric filenames
        if [[ "$test_num" =~ ^[0-9]+$ ]]; then
            run_test "$test_num"
        fi
    done

else

    # Run selected tests
    for arg in "$@"; do

        # Range, e.g. 3-7
        if [[ "$arg" =~ ^([0-9]+)-([0-9]+)$ ]]; then

            start="${BASH_REMATCH[1]}"
            end="${BASH_REMATCH[2]}"

            if [ "$start" -gt "$end" ]; then
                echo -e "${RED}Error: Invalid range '$arg'${NC}"
                exit 1
            fi

            for ((i=start; i<=end; i++)); do
                run_test "$i"
            done

        # Single test number
        elif [[ "$arg" =~ ^[0-9]+$ ]]; then

            run_test "$arg"

        # Invalid argument
        else

            echo -e "${RED}Error: Invalid test specification '$arg'${NC}"
            echo
            echo "Usage:"
            echo "  ./run_tests.sh          # Run all tests"
            echo "  ./run_tests.sh 3       # Run test 3"
            echo "  ./run_tests.sh 1 4 7   # Run tests 1, 4, 7"
            echo "  ./run_tests.sh 1-5     # Run tests 1 through 5"
            echo "  ./run_tests.sh 1 4-7   # Run test 1 and tests 4 through 7"
            exit 1

        fi

    done

fi


# ============================================================
# Summary
# ============================================================

echo "=========================================="
echo -e "Summary: Total: $TOTAL | ${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC}"
echo "=========================================="


# ============================================================
# Cleanup
# ============================================================

# Remove temporary outputs only if every executed test passed
if [ $FAILED -eq 0 ] && [ $TOTAL -gt 0 ]; then
    rm -rf "$TEMP_DIR"
fi