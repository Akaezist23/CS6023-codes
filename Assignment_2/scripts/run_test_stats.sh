#!/usr/bin/env bash
# =============================================================================
#  run_tests_stats.sh -- run run_tests.sh N times and report timing stats
#                         (mean, stddev, min, max) per test case.
#
#  This script does NOT reimplement any testing logic. It just repeatedly
#  invokes the existing scripts/run_tests.sh, captures its stdout (which
#  prints one "input<N>  RESULT  <ms> ms  detail" line per test case), and
#  aggregates the timings across runs.
#
#  Usage:
#     ./scripts/run_tests_stats.sh -n 10                 # run every submit/*.cu, 10x
#     ./scripts/run_tests_stats.sh -n 20 -s submit/x.cu  # run one file, 20x
#     ./scripts/run_tests_stats.sh -n 10 -c 3            # only test case 3, 10x
#     ./scripts/run_tests_stats.sh -n 10 -K              # keep the raw per-run logs
#
#  Any flag run_tests.sh understands (-s, -c, -v, -k) can be passed through
#  unchanged; this script only adds -n (required) and -K (keep raw logs).
#
#  Output: a table printed to screen, and also written to stats.txt in the
#  assignment folder (next to results.txt).
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_TESTS="$ROOT/scripts/run_tests.sh"
STATS_OUT="${A2_STATS_OUT:-$ROOT/stats.txt}"

N=""
KEEP_LOGS=0
PASSTHRU=()

usage() { awk 'NR>1 && /^#/ { sub(/^#[[:space:]]?/, ""); print; next } NR>1 { exit }' "$0"; }

if [ ! -x "$RUN_TESTS" ] && [ ! -f "$RUN_TESTS" ]; then
    echo "ERROR: could not find $RUN_TESTS" >&2
    exit 2
fi

# Parse just -n / -K ourselves; forward everything else verbatim to run_tests.sh.
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--num-runs)
            [ $# -lt 2 ] && { echo "ERROR: $1 needs a value." >&2; exit 2; }
            N="$2"; shift 2 ;;
        -K|--keep-logs)
            KEEP_LOGS=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            PASSTHRU+=("$1"); shift ;;
    esac
done

if [ -z "$N" ] || ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "ERROR: -n <positive integer> is required (number of repetitions)." >&2
    usage
    exit 2
fi

if [ -t 1 ]; then
    GRN=$'\033[0;32m'; BLD=$'\033[1m'; OFF=$'\033[0m'
else
    GRN=""; BLD=""; OFF=""
fi

WORK="$ROOT/.work_stats"
rm -rf "$WORK"
mkdir -p "$WORK"
cleanup() { [ "$KEEP_LOGS" -eq 0 ] && rm -rf "$WORK"; }
trap cleanup EXIT

echo "${BLD}Running '$(basename "$RUN_TESTS") ${PASSTHRU[*]:-}' x $N ...${OFF}"

for ((i = 1; i <= N; i++)); do
    printf "  run %d/%d ...\r" "$i" "$N"
    log="$WORK/run_${i}.log"
    # Redirecting to a file means run_tests.sh's own `[ -t 1 ]` check is
    # false, so it emits plain (uncoloured) text -- easy to parse below.
    bash "$RUN_TESTS" "${PASSTHRU[@]}" > "$log" 2>&1
done
echo
echo "${GRN}Done. Aggregating timings...${OFF}"
echo

# -----------------------------------------------------------------------
# Parse every log. Lines of interest look like (module: run_tests.sh):
#   === submit/foo.cu ===
#   input1       PASS      12.34 ms     output matches
#   input2       FAIL      9.87 ms      output differs (first diff at ...)
#   input3       CRASH     -            exit code 1
#   input4       BUILD FAILED : 0/6 test cases passed          (no per-test lines follow)
# -----------------------------------------------------------------------
STATS_TSV="$WORK/stats.tsv"

cat "$WORK"/run_*.log | awk '
    /^=== .* ===$/ {
        src = $0
        sub(/^=== /, "", src)
        sub(/ ===$/, "", src)
        next
    }
    /^BUILD FAILED/ { build_fail[src]++; next }
    $1 ~ /^input[0-9]+$/ {
        key = src SUBSEP $1
        total[key]++
        status[key, $2]++
        if ($3 != "-" && $4 == "ms") {
            v = $3 + 0
            n[key]++
            sum[key]   += v
            sumsq[key] += v*v
            if (!(key in minv) || v < minv[key]) minv[key] = v
            if (!(key in maxv) || v > maxv[key]) maxv[key] = v
        }
        next
    }
    END {
        for (k in total) {
            split(k, parts, SUBSEP)
            s = parts[1]; t = parts[2]
            cnt  = n[k] + 0
            mean = (cnt > 0) ? sum[k] / cnt : 0
            var  = 0
            if (cnt > 1) var = (sumsq[k] - sum[k]*sum[k]/cnt) / (cnt - 1)
            sd = (var > 0) ? sqrt(var) : 0
            p  = status[k, "PASS"] + 0
            printf "%s\t%s\t%d\t%d\t%.4f\t%.4f\t%.4f\t%.4f\n", \
                   s, t, total[k], p, mean, sd, (cnt>0?minv[k]:0), (cnt>0?maxv[k]:0)
        }
        for (s in build_fail)
            printf "%s\tBUILD_FAILED\t%d\t0\t0\t0\t0\t0\n", s, build_fail[s]
    }
' > "$STATS_TSV"

if [ ! -s "$STATS_TSV" ]; then
    echo "No timing data was found -- did every run fail to build?"
    echo "Raw logs are in $WORK (re-run with -K to keep them for inspection)."
    exit 1
fi

# Sort by source, then numerically by the test-case number embedded in
# "input<N>" (falls back to lexical order for the BUILD_FAILED row).
sort_key() { echo "$1" | sed -E 's/^input([0-9]+)$/\1/'; }

{
    printf "%-28s %-14s %8s %8s %12s %12s %10s %10s\n" \
           "SOURCE" "TESTCASE" "RUNS" "PASSES" "MEAN(ms)" "STDDEV(ms)" "MIN(ms)" "MAX(ms)"
    printf '%s\n' "--------------------------------------------------------------------------------------------------------------"
} | tee "$STATS_OUT"

awk -F'\t' '{ printf "%s\t%013d\t%s\n", $1, ($2 ~ /^input[0-9]+$/ ? substr($2,6)+0 : 999999), $0 }' "$STATS_TSV" \
    | sort -t $'\t' -k1,1 -k2,2n \
    | cut -f3- \
    | while IFS=$'\t' read -r src test runs passes mean sd mn mx; do
        if [ "$test" = "BUILD_FAILED" ]; then
            printf "%-28s %-14s %8s %8s %12s %12s %10s %10s\n" \
                   "$src" "BUILD FAILED" "$runs" "-" "-" "-" "-" "-" | tee -a "$STATS_OUT"
        else
            printf "%-28s %-14s %8s %8s %12.4f %12.4f %10.4f %10.4f\n" \
                   "$src" "$test" "$runs" "$passes" "$mean" "$sd" "$mn" "$mx" | tee -a "$STATS_OUT"
        fi
    done

echo
echo "Stats written to: $STATS_OUT"
[ "$KEEP_LOGS" -eq 1 ] && echo "Raw per-run logs kept in: $WORK"