#!/usr/bin/env python3
# Converts the text table a compare_*.sh script prints into a CSV next to it, for plotting
# (e.g. pgfplots: \addplot table [col sep=comma, x=N, y=base] {file.csv};).
# Usage: python3 table_to_csv.py <sizes|blocks|depth|resolution|all> results/foo.txt  -> results/foo.csv
# Speedups lose their "x"; missing values ("-", "fail") become nan. Column names use "_"
# instead of "@" (e.g. timed_32x32) and blocks get a "b" prefix (b32x32).
import re
import sys


def num(tok):
    tok = tok.rstrip("x") if re.fullmatch(r"[\d.]+x", tok) else tok
    return "nan" if tok in ("-", "fail") else tok


def name(label):
    return label.replace("@", "_")


def convert(kind, lines):
    rows, header = [], None
    for i, line in enumerate(lines):
        t = line.split()
        if not t:
            continue
        if kind == "sizes":
            if t[0] == "N":
                header = ["N", "base", "opt", "speedup", "base_ms_per_sphere", "opt_ms_per_sphere"]
            elif header and re.fullmatch(r"\d+", t[0]) and len(t) == 6:
                rows.append([num(x) for x in t])
        elif kind == "blocks":
            if t[0] == "N":
                header = ["N"] + ["b" + s for s in t[1:]]
            elif header and re.fullmatch(r"\d+", t[0]):
                rows.append([num(x) for x in t])
        elif kind == "depth":
            if t[0] == "depth":
                header = ["depth"] + [name(s) for s in t[1:-2]] + ["speedup", "step"]
            elif header and re.fullmatch(r"\d+", t[0]):
                rows.append([num(x) for x in t])
        elif kind == "resolution":
            if t[0] == "WxH":
                labels = [name(s) for s in t[2:-1]]
                header = (["W", "H", "Mpix"] + labels + ["speedup"]
                          + [l + "_ms_per_mpix" for l in labels])
            elif header and re.fullmatch(r"\d+x\d+", t[0]):
                per = lines[i + 1].split()[1:] if i + 1 < len(lines) else []   # the "/Mpix" row
                w, h = t[0].split("x")
                rows.append([w, h] + [num(x) for x in t[1:]] + [num(x) for x in per])
        elif kind == "all":
            if t[0] == "version":
                header = ["version", "block", "mean", "std", "min", "max", "vs_ref", "vs_prev"]
            elif header and len(t) >= 3:
                vals = [num(x) for x in t[2:]] + ["nan"] * 6
                rows.append([t[0], t[1]] + vals[:6])
    if header is None:
        sys.exit(f"no {kind} table found")
    return [header] + rows


kind, path = sys.argv[1], sys.argv[2]
with open(path) as f:
    table = convert(kind, f.read().splitlines())
out = re.sub(r"\.txt$", "", path) + ".csv"
with open(out, "w") as f:
    f.writelines(",".join(r) + "\n" for r in table)
print(f"Wrote {out} ({len(table) - 1} rows)")
