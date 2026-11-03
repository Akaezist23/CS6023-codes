#!/usr/bin/env python3
# Compare two PPM (P6) images pixel by pixel.
# Usage: python3 compare_images.py reference.ppm candidate.ppm [diff.ppm]
# Exit code 0 if identical, 1 otherwise.
# With diff.ppm, also writes an image of the reference dimmed to 25 %, with every
# differing pixel in red (brightness = size of the difference, at least 50 %).
import sys


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    # Header: "P6" width height maxval, separated by whitespace, then one whitespace byte.
    fields, pos = [], 0
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(data[start:pos])
    if fields[0] != b"P6":
        sys.exit(f"{path}: not a P6 PPM")
    w, h = int(fields[1]), int(fields[2])
    return w, h, data[pos + 1:pos + 1 + w * h * 3]


(w1, h1, a), (w2, h2, b) = read_ppm(sys.argv[1]), read_ppm(sys.argv[2])
if (w1, h1) != (w2, h2):
    sys.exit(f"size mismatch: {w1}x{h1} vs {w2}x{h2}")

diff_px, max_diff, first = 0, 0, None
for i in range(0, len(a), 3):
    d = max(abs(a[i] - b[i]), abs(a[i + 1] - b[i + 1]), abs(a[i + 2] - b[i + 2]))
    if d:
        diff_px += 1
        max_diff = max(max_diff, d)
        if first is None:
            first = (i // 3 % w1, i // 3 // w1)

if len(sys.argv) > 3:
    out = bytearray(x // 4 for x in a)
    for i in range(0, len(a), 3):
        d = max(abs(a[i] - b[i]), abs(a[i + 1] - b[i + 1]), abs(a[i + 2] - b[i + 2]))
        if d:
            out[i:i + 3] = bytes((max(d, 128), 0, 0))
    with open(sys.argv[3], "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (w1, h1) + bytes(out))

total = w1 * h1
if diff_px == 0:
    print(f"IDENTICAL ({w1}x{h1})")
    sys.exit(0)
print(f"DIFFERENT: {diff_px}/{total} pixels ({100 * diff_px / total:.3f}%), "
      f"max channel diff {max_diff}/255, first at (x={first[0]}, y={first[1]})")
sys.exit(1)
