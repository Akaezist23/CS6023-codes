#!/usr/bin/env python3
"""
generate_input.py -- generates a random test-case input file in the
format expected by pipeline.cu (see CS6023 Assignment 1 spec).
Vectorized with numpy so it stays fast even at the max allowed size
(H, W, Hr, Wr up to 1024).

Usage:
    python3 generate_input.py H W Hr Wr Hc Wc mean std [seed] > input.txt

Example (max size):
    python3 generate_input.py 1024 1024 1024 1024 1000 1000 0.45 0.22 42 > input.txt

Constraints enforced (per spec):
    1 <= H, W       <= 1024
    1 <= Hr, Wr     <= 1024
    1 <= Hc <= Hr,  1 <= Wc <= Wr
    0.0 <= mean <= 1.0
    0.01 <= std <= 2.0
"""

import sys
import numpy as np


def main():
    if len(sys.argv) not in (9, 10):
        sys.stderr.write(
            "Usage: python3 generate_input.py H W Hr Wr Hc Wc mean std [seed]\n"
        )
        sys.exit(1)

    H, W, Hr, Wr, Hc, Wc = (int(a) for a in sys.argv[1:7])
    mean, std = float(sys.argv[7]), float(sys.argv[8])
    seed = int(sys.argv[9]) if len(sys.argv) == 10 else None

    assert 1 <= H <= 1024 and 1 <= W <= 1024, "H, W out of range"
    assert 1 <= Hr <= 1024 and 1 <= Wr <= 1024, "Hr, Wr out of range"
    assert 1 <= Hc <= Hr, "Hc must be in [1, Hr]"
    assert 1 <= Wc <= Wr, "Wc must be in [1, Wr]"
    assert 0.0 <= mean <= 1.0, "mean out of range"
    assert 0.01 <= std <= 2.0, "std out of range"

    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(H * W * 3,), dtype=np.int32)

    header = (
        f"{H} {W}\n"
        f"{Hr} {Wr}\n"
        f"{Hc} {Wc}\n"
        f"{mean:.6f} {std:.6f}\n"
    )

    # Reshape to (H, W*3) so each row of text = one image row (matches the
    # style of the sample input, though the parser only cares about token
    # order, not line breaks).
    rows = pixels.reshape(H, W * 3)

    out = sys.stdout
    out.write(header)
    # np.savetxt is a fast, C-backed way to dump a big integer matrix as
    # whitespace-separated text.
    np.savetxt(out, rows, fmt="%d")


if __name__ == "__main__":
    main()
