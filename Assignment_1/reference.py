#!/usr/bin/env python3
"""
reference.py -- sequential (CPU) reference implementation of the
"Wizard's Lens" preprocessing pipeline from the CS6023 GPU Programming
assignment.

This is ONLY meant to be used as a correctness oracle for testing your
CUDA implementation. It reads the exact same input format described in
the assignment PDF and prints the exact same output format, so its
output can be diffed / compared against your pipeline.cu's output.

Implemented with numpy purely for speed on large (up to 1024x1024)
inputs -- the formulas are identical, element-for-element, to a naive
triple-nested-loop implementation. This has been checked against the
worked example in the assignment PDF and against a pure-Python
loop-based version for equivalence.

Usage:
    python3 reference.py < input.txt > expected_output.txt
    python3 reference.py input.txt > expected_output.txt
"""

import sys
import io
import numpy as np


def read_tokens(path=None):
    if path is None:
        data = sys.stdin.buffer.read()
    else:
        with open(path, "rb") as f:
            data = f.read()
    return data.split()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    tok = read_tokens(path)
    idx = 0

    def next_int():
        nonlocal idx
        v = int(tok[idx])
        idx += 1
        return v

    def next_float():
        nonlocal idx
        v = float(tok[idx])
        idx += 1
        return v

    H = next_int()
    W = next_int()
    Hr = next_int()
    Wr = next_int()
    Hc = next_int()
    Wc = next_int()
    mean = next_float()
    std = next_float()

    n_pixels = H * W
    pixel_tokens = tok[idx: idx + 3 * n_pixels]
    idx += 3 * n_pixels

    # ---- parse raw RGB, row-major, channel-interleaved R G B R G B ... ----
    rgb = np.array(pixel_tokens, dtype=np.float64).reshape(H, W, 3)
    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # ---- Stage 1: Grayscale ----
    # gray(y,x) = 0.299 R + 0.587 G + 0.114 B   (kept as float, not rounded)
    gray = 0.299 * R + 0.587 * G + 0.114 * B  # shape (H, W)

    # ---- Stage 2: Bilinear resize (align-corners convention) ----
    if Hr > 1:
        sy = (H - 1) / (Hr - 1)
    else:
        sy = 0.0
    if Wr > 1:
        sx = (W - 1) / (Wr - 1)
    else:
        sx = 0.0

    oy = np.arange(Hr, dtype=np.float64)
    ox = np.arange(Wr, dtype=np.float64)

    fy = oy * sy  # shape (Hr,)
    fx = ox * sx  # shape (Wr,)

    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)

    wy = (fy - y0).reshape(Hr, 1)   # shape (Hr, 1)
    wx = (fx - x0).reshape(1, Wr)   # shape (1, Wr)

    # Gather the 4 corners for every output pixel via broadcasting.
    # gray[y0][:, x0] etc. give shape (Hr, Wr).
    g00 = gray[np.ix_(y0, x0)]
    g01 = gray[np.ix_(y0, x1)]
    g10 = gray[np.ix_(y1, x0)]
    g11 = gray[np.ix_(y1, x1)]

    top = g00 * (1 - wx) + g01 * wx
    bottom = g10 * (1 - wx) + g11 * wx
    resized = top * (1 - wy) + bottom * wy  # shape (Hr, Wr)

    # ---- Stage 3: Center crop ----
    off_y = (Hr - Hc) // 2
    off_x = (Wr - Wc) // 2
    cropped = resized[off_y: off_y + Hc, off_x: off_x + Wc]

    # ---- Stage 4: Normalize ----
    out = (cropped / 255.0 - mean) / std

    # ---- Print in the required format ----
    sys.stdout.write(f"{Hc} {Wc}\n")
    buf = io.StringIO()
    np.savetxt(buf, out, fmt="%.6f")
    sys.stdout.write(buf.getvalue())


if __name__ == "__main__":
    main()
