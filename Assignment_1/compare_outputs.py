#!/usr/bin/env python3
"""
compare_outputs.py -- compares your CUDA program's output against the
reference (expected) output, using the exact tolerance rule from the
assignment:

    |a - e| <= 1e-3 + 1e-4 * |e|

Usage:
    python3 compare_outputs.py actual.txt expected.txt
Exit code: 0 if PASS, 1 if FAIL.
"""

import sys


def load(path):
    with open(path) as f:
        tok = f.read().split()
    if len(tok) < 2:
        raise ValueError(f"{path}: file too short / empty")
    h, w = int(tok[0]), int(tok[1])
    vals = list(map(float, tok[2:]))
    expected_count = h * w
    if len(vals) != expected_count:
        raise ValueError(
            f"{path}: header says {h}x{w} ({expected_count} values) "
            f"but found {len(vals)} value(s)"
        )
    return h, w, vals


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: python3 compare_outputs.py actual.txt expected.txt\n")
        sys.exit(2)

    actual_path, expected_path = sys.argv[1], sys.argv[2]

    try:
        ah, aw, avals = load(actual_path)
    except Exception as e:
        print(f"FAIL: could not parse actual output ({actual_path}): {e}")
        sys.exit(1)

    try:
        eh, ew, evals = load(expected_path)
    except Exception as e:
        print(f"FAIL: could not parse expected output ({expected_path}): {e}")
        sys.exit(1)

    if (ah, aw) != (eh, ew):
        print(f"FAIL: dimension mismatch. actual={ah}x{aw} expected={eh}x{ew}")
        sys.exit(1)

    n_bad = 0
    max_err = 0.0
    worst = None
    for i, (a, e) in enumerate(zip(avals, evals)):
        tol = 1e-3 + 1e-4 * abs(e)
        err = abs(a - e)
        if err > tol:
            n_bad += 1
            if worst is None or err - tol > worst[0]:
                worst = (err - tol, i, a, e, err, tol)
        if err > max_err:
            max_err = err

    if n_bad == 0:
        print(f"PASS  ({ah}x{aw} values, max abs error = {max_err:.6e})")
        sys.exit(0)
    else:
        row = worst[1] // aw
        col = worst[1] % aw
        print(
            f"FAIL: {n_bad}/{len(evals)} value(s) out of tolerance.\n"
            f"  Worst mismatch at (row={row}, col={col}): "
            f"actual={worst[2]:.6f} expected={worst[3]:.6f} "
            f"err={worst[4]:.6e} tol={worst[5]:.6e}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
