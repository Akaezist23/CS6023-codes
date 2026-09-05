#!/usr/bin/env python3
"""
reference.py -- sequential (CPU) reference implementation of the
"Wizard's Lens" preprocessing pipeline from the CS6023 GPU Programming
assignment (grayscale -> bilinear resize -> center crop -> normalize).

Vectorized with numpy so it stays fast even at the max size (1024x1024).
This is ONLY meant to be used as a correctness oracle for testing your
CUDA implementation -- it reads the exact input format from the PDF and
prints the exact output format, so it can be diffed against your
pipeline.cu's output.

Usage:
    python3 reference.py < input.txt > expected_output.txt
    python3 reference.py input.txt > expected_output.txt
"""

import sys
import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        data = sys.stdin.read()
    else:
        with open(path, "r") as f:
            data = f.read()

    # Parse every whitespace-separated token as float64 in one shot (fast
    # even for ~3M tokens at 1024x1024x3). We cast header fields to int
    # where appropriate.
    tok = data.split()
    arr = np.array(tok, dtype=np.float64)

    idx = 0
    H = int(arr[idx]); idx += 1
    W = int(arr[idx]); idx += 1
    Hr = int(arr[idx]); idx += 1
    Wr = int(arr[idx]); idx += 1
    Hc = int(arr[idx]); idx += 1
    Wc = int(arr[idx]); idx += 1
    mean = arr[idx]; idx += 1
    std = arr[idx]; idx += 1

    pixels = arr[idx: idx + H * W * 3]
    if pixels.size != H * W * 3:
        raise ValueError(
            f"Expected {H * W * 3} pixel values, found {pixels.size}"
        )
    rgb = pixels.reshape(H, W, 3)  # channel-interleaved, row-major

    # ---- Stage 1: Grayscale ----
    # gray(y,x) = 0.299 R + 0.587 G + 0.114 B  (kept as float, not rounded)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray = (rgb * weights).sum(axis=2)  # shape (H, W), float64

    # ---- Stage 2: Bilinear resize (align-corners convention) ----
    sy = (H - 1) / (Hr - 1) if Hr > 1 else 0.0
    sx = (W - 1) / (Wr - 1) if Wr > 1 else 0.0

    oy = np.arange(Hr, dtype=np.float64)
    ox = np.arange(Wr, dtype=np.float64)

    fy = oy * sy                      # (Hr,)
    fx = ox * sx                      # (Wr,)

    y0 = np.floor(fy).astype(np.int64)
    x0 = np.floor(fx).astype(np.int64)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)

    wy = (fy - y0).reshape(Hr, 1)     # (Hr, 1) broadcast over columns
    wx = (fx - x0).reshape(1, Wr)     # (1, Wr) broadcast over rows

    # Gather the 4 corner grids via outer indexing: shape (Hr, Wr)
    g_y0x0 = gray[np.ix_(y0, x0)]
    g_y0x1 = gray[np.ix_(y0, x1)]
    g_y1x0 = gray[np.ix_(y1, x0)]
    g_y1x1 = gray[np.ix_(y1, x1)]

    top = g_y0x0 * (1 - wx) + g_y0x1 * wx
    bottom = g_y1x0 * (1 - wx) + g_y1x1 * wx
    resized = top * (1 - wy) + bottom * wy   # (Hr, Wr)

    # ---- Stage 3: Center crop ----
    off_y = (Hr - Hc) // 2
    off_x = (Wr - Wc) // 2
    cropped = resized[off_y:off_y + Hc, off_x:off_x + Wc]

    # ---- Stage 4: Normalize ----
    out = (cropped / 255.0 - mean) / std

    # ---- Print in the required format ----
    buf = [f"{Hc} {Wc}"]
    fmt_row = lambda row: " ".join(f"{v:.6f}" for v in row)
    buf.extend(fmt_row(row) for row in out)
    sys.stdout.write("\n".join(buf) + "\n")


if __name__ == "__main__":
    main()
