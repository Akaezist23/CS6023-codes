#!/usr/bin/env python3
# Merges the per-N compare_all CSVs that sweep_all.sh writes into one CSV for plotting:
#   N, <version>_<block> (mean ms), <version>_<block>_std, <version>_<block>_speedup (vs first version),
#   <version>_<block>_us_per_sph: cost of each extra sphere since the previous N, in microseconds
#   ((time - previous time) / (N - previous N); nan in the first row)
# Usage: python3 merge_sweep.py results/sweep_all results/sweep_all.csv
import csv
import glob
import os
import re
import sys

src_dir, out = sys.argv[1], sys.argv[2]
rows, labels = {}, []
for path in glob.glob(os.path.join(src_dir, "compare_all_N*.csv")):
    n = int(re.search(r"_N(\d+)", os.path.basename(path)).group(1))
    with open(path) as f:
        for r in csv.DictReader(f):
            label = f"{r['version']}_{r['block']}"
            if label not in labels:
                labels.append(label)
            rows.setdefault(n, {})[label] = (r["mean"], r["std"])
if not rows:
    sys.exit(f"no compare_all_N*.csv in {src_dir}")

ref = labels[0]
header = ["N"] + [f"{l}{s}" for l in labels for s in ("", "_std", "_speedup", "_us_per_sph")]
with open(out, "w") as f:
    f.write(",".join(header) + "\n")
    prev = None
    for n in sorted(rows):
        line = [str(n)]
        for l in labels:
            mean, std = rows[n].get(l, ("nan", "nan"))
            r = rows[n].get(ref, ("nan",))[0]
            ok = "nan" not in (mean, r) and float(mean) > 0
            pm = rows[prev].get(l, ("nan",))[0] if prev is not None else "nan"
            dc = f"{(float(mean) - float(pm)) / (n - prev) * 1e3:.2f}" if "nan" not in (mean, pm) else "nan"
            line += [mean, std, f"{float(r) / float(mean):.3f}" if ok else "nan", dc]
        prev = n
        f.write(",".join(line) + "\n")
print(f"Wrote {out} ({len(rows)} sizes, {len(labels)} versions)")
