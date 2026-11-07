#!/usr/bin/env python3
# Extracts the stall breakdown and the roofline numbers from ncu reports into CSVs, for plotting.
# Usage: NCU=/path/to/ncu python3 ncu_to_csv.py OUT_PREFIX profile_dir1 profile_dir2 ...
#   Each profile_dir holds an ncu_report.ncu-rep (as written by profile.sh); its folder name
#   (e.g. split_8x8_N5000) becomes the "version" column. Rows keep the order of the arguments.
# Writes
#   OUT_PREFIX_stalls.csv     one row per version
#   OUT_PREFIX_roofline.csv   one row per version (performance, intensity and bandwidth per level, peaks)
#   OUT_PREFIX_levels.csv     one row per memory level (L1, L2, DRAM), columns ai_<version>, gflops_<version>:
#                             each version is one pgfplots \addplot of three points
#
# Stalls: for each reason, the average number of cycles a warp spends stalled on it per issued
#   instruction (ncu "Warp State" chart, without "selected", which is the issue cycle itself), and the
#   same multiplied by the issued instructions, in billions of warp-cycles (_gcycles: total waiting,
#   comparable across versions with different instruction counts). "other" sums the small reasons.
# Roofline: same definitions as ncu's hierarchical single-precision roofline chart
#   (sections/SpeedOfLight_HierarchicalSingleRooflineChart.section): FLOP = fadd + fmul + 2 ffma;
#   bytes per level = L1 lsu writeback cycles x 128, L2 lts2xbar cycles x 32, DRAM bytes; peaks from
#   the *.peak_sustained metrics times each unit's clock (ncu locks the clocks at base).
import csv
import io
import os
import subprocess
import sys

NCU = os.environ.get("NCU", "ncu")
MAIN = ["long_scoreboard", "wait", "not_selected", "math_pipe_throttle", "lg_throttle",
        "branch_resolving", "no_instruction"]
SMALL = ["dispatch_stall", "short_scoreboard", "mio_throttle", "barrier", "membar", "drain",
         "imc_miss", "misc", "sleeping", "tex_throttle"]
LEVELS = ["l1", "l2", "dram"]


def raw_metrics(rep, base_units):
    # base units (bytes, cycles, ns) for the roofline; the default units keep full precision for ratios
    cmd = [NCU, "--import", rep, "--page", "raw", "--csv"] + (["--print-units", "base"] if base_units else [])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    names, values = rows[0], rows[2]   # row 1 holds the units
    m = {}
    for n, v in zip(names, values):
        try:
            m[n] = float(v.replace(",", ""))
        except ValueError:
            pass
    return m


def write(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        f.writelines(",".join(r) + "\n" for r in rows)


prefix, dirs = sys.argv[1], sys.argv[2:]
versions, stall_rows, roof_rows, levels = [], [], [], {}
for d in dirs:
    version = os.path.basename(os.path.normpath(d))
    versions.append(version)
    rep = os.path.join(d, "ncu_report.ncu-rep")
    m, r = raw_metrics(rep, True), raw_metrics(rep, False)

    inst = m["inst_executed"]
    ratio = lambda s: r.get(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio", 0.0)
    per_issue = {s: ratio(s) for s in MAIN}
    per_issue["other"] = sum(ratio(s) for s in SMALL)
    stall_rows.append([version, f"{inst:.0f}", f"{sum(per_issue.values()):.3f}"]
                      + [f"{per_issue[s]:.3f}" for s in per_issue]
                      + [f"{per_issue[s] * inst / 1e9:.3f}" for s in per_issue]
                      + [f"{sum(per_issue.values()) * inst / 1e9:.3f}"])

    t = m["gpu__time_duration.sum"]                       # nanoseconds (base units)
    op = "smsp__sass_thread_inst_executed_op_{}_pred_on.sum.per_cycle_elapsed"
    perf = ((m[op.format("fadd")] + m[op.format("fmul")] + 2 * m[op.format("ffma")])
            * m["smsp__cycles_elapsed.avg.per_second"])   # FLOP/s, as in ncu's roofline
    bw = {"l1": m["l1tex__lsu_writeback_active_mem_lg.sum.per_second"] * 128,
          "l2": m["lts__lts2xbar_cycles_active.sum.per_second"] * 32,
          "dram": m["dram__bytes.sum.per_second"]}
    peak = {"fp32": 2 * m["sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained"]
                    * m["sm__cycles_elapsed.avg.per_second"],
            "l1": 128 * m["l1tex__lsu_writeback_active_mem_lg.sum.peak_sustained"]
                  * m["l1tex__cycles_elapsed.avg.per_second"],
            "l2": 32 * m["lts__lts2xbar_cycles_active.sum.peak_sustained"]
                  * m["lts__cycles_elapsed.avg.per_second"],
            "dram": m["dram__bytes.sum.peak_sustained"] * m["dram__cycles_elapsed.avg.per_second"]}
    ai = {k: perf / bw[k] if bw[k] > 0 else float("nan") for k in LEVELS}
    roof_rows.append([version, f"{t / 1e6:.2f}", f"{perf / 1e9:.1f}"]
                     + [f"{ai[k]:.3f}" for k in LEVELS]
                     + [f"{bw[k] / 1e9:.2f}" for k in LEVELS]
                     + [f"{peak[k] / 1e9:.1f}" for k in ("fp32", "l1", "l2", "dram")])
    levels[version] = (ai, perf / 1e9)

write(f"{prefix}_stalls.csv",
      ["version", "issued_inst", "cycles_per_issue"] + MAIN + ["other"]
      + [f"{s}_gcycles" for s in MAIN + ["other"]] + ["total_gcycles"], stall_rows)
write(f"{prefix}_roofline.csv",
      ["version", "time_ms", "gflops"] + [f"ai_{k}" for k in LEVELS] + [f"bw_{k}_gbs" for k in LEVELS]
      + ["peak_fp32_gflops", "peak_l1_gbs", "peak_l2_gbs", "peak_dram_gbs"], roof_rows)
write(f"{prefix}_levels.csv",
      ["level"] + [f"{c}_{v}" for v in versions for c in ("ai", "gflops")],
      [[k.upper()] + [f"{x}" for v in versions for x in (f"{levels[v][0][k]:.3f}", f"{levels[v][1]:.1f}")]
       for k in LEVELS])
print(f"Wrote {prefix}_stalls.csv, _roofline.csv, _levels.csv ({len(dirs)} profiles)")
