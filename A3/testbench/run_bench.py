#!/usr/bin/env python3
"""A3 delta-stepping testbench.

Builds the generator, the CPU reference and the solution, generates heavy and
edge-case graphs, and runs the solution several times per test to check
correctness (vs. the reference) and determinism (all runs identical).

  python3 run_bench.py --sol CS23B078.cu                 # default suite
  python3 run_bench.py --sol CS23B078.cu --only a_k1 s_delta0
  python3 run_bench.py --sol CS23B078.cu --huge          # adds E = 300M tests
  python3 run_bench.py --sol CS23B078.cu --runs 5 --timeout 900
"""
import argparse
import filecmp
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# name: (N, E, S, delta, K, type, wmin, wmax)
TESTS = {
    # ---- static ----
    "s_rand_big":    (2_000_000, 40_000_000, 3,   50,     -1, "rand",   0, 1000),
    "s_hub":         (1_000_000, 20_000_000, 3,  100,     -1, "hub",    1, 1000),
    "s_delta0":      (  200_000,  2_000_000, 2,    0,     -1, "rand",   0,   20),
    "s_delta1_chain":(  100_000,    400_000, 2,    1,     -1, "chain",  1,   10),
    "s_bigdelta":    (2_000_000, 30_000_000, 2, 1000,     -1, "rand",   0, 1000),
    "s_sparse":      (1_000_000,  5_000_000, 3,   10,     -1, "sparse", 0,  100),
    "s_allzero":     (  500_000,  5_000_000, 2,    5,     -1, "rand",   0,    0),
    "s_20src":       (  100_000,  1_000_000, 20,   7,     -1, "rand",   0, 1000),
    # ---- adaptive ----
    "a_rand_big":    (2_000_000, 40_000_000, 3,   -1,  65536, "rand",   0, 1000),
    "a_hub_plateau": (1_000_000, 20_000_000, 3,   -1,   1024, "hub",    0,    5),
    "a_kmax":        (2_000_000, 30_000_000, 2,   -1, 524288, "rand",   1, 1000),
    "a_k1":          (   20_000,    200_000, 2,   -1,      1, "rand",   0,    3),
    "a_chain":       (  200_000,  1_000_000, 2,   -1,     16, "chain",  1,  100),
    "a_sparse":      (1_000_000,  5_000_000, 3,   -1,   4096, "sparse", 0,   10),
    "a_zero_one":    (  500_000,  5_000_000, 2,   -1,     64, "rand",   0,    1),
    "a_20src":       (  100_000,  1_000_000, 20,  -1,    256, "hub",    0, 1000),
    # ---- edge cases ----
    "e_single_s":    (        1,          0, 2,    3,     -1, "rand",   0,    5),
    "e_single_a":    (        1,          0, 2,   -1,      5, "rand",   0,    5),
    "e_selfloops_a": (    1_000,     20_000, 20,  -1,      7, "hub",    0,   10),
    "e_k_gt_n":      (    1_000,      5_000, 3,   -1, 524288, "rand",   0, 1000),
}

HUGE = {
    "h_static":      (2_000_000, 300_000_000, 1,  100,    -1, "rand",   0, 1000),
    "h_adaptive":    (2_000_000, 300_000_000, 1,   -1, 65536, "rand",   0, 1000),
}

GREEN, RED, YELLOW, NC = "\033[0;32m", "\033[0;31m", "\033[1;33m", "\033[0m"


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def first_diff(a, b, limit=5):
    """Return (count of differing lines, list of first few diffs), ignoring blank lines."""
    def lines(path):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if s:
                    yield i, s
    diffs, count = [], 0
    ia, ib = lines(a), lines(b)
    while True:
        x, y = next(ia, None), next(ib, None)
        if x is None and y is None:
            break
        xs = x[1] if x else "<EOF>"
        ys = y[1] if y else "<EOF>"
        if xs != ys:
            count += 1
            if len(diffs) < limit:
                diffs.append((x[0] if x else y[0], xs, ys))
    return count, diffs


def same(a, b):
    if filecmp.cmp(a, b, shallow=False):
        return True
    return first_diff(a, b, limit=0)[0] == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", required=True, help="path to the .cu solution")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600, help="seconds per run")
    ap.add_argument("--only", nargs="*", help="run only these tests")
    ap.add_argument("--huge", action="store_true", help="also run E = 300M tests")
    ap.add_argument("--workdir", default=os.path.join(HERE, "work"))
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    gen = os.path.join(args.workdir, "gen")
    ref = os.path.join(args.workdir, "ref")
    sol = os.path.join(args.workdir, "main")
    sh(["g++", "-O2", "-std=c++17", "-o", gen, os.path.join(HERE, "gen.cpp")])
    sh(["g++", "-O2", "-std=c++17", "-o", ref, os.path.join(HERE, "ref.cpp")])
    sh(["nvcc", args.sol, "-o", sol])  # same command the graders use

    tests = dict(TESTS)
    if args.huge:
        tests.update(HUGE)
    if args.only:
        missing = [t for t in args.only if t not in tests]
        if missing:
            sys.exit(f"unknown tests: {missing} (huge tests need --huge)")
        tests = {t: tests[t] for t in args.only}

    results = []
    for seed, (name, (N, E, S, delta, K, typ, wmin, wmax)) in enumerate(tests.items(), 1):
        print(f"\n=== {name}: N={N:,} E={E:,} S={S} delta={delta} K={K} type={typ} w=[{wmin},{wmax}]", flush=True)
        inp = os.path.join(args.workdir, name + ".in")
        exp = os.path.join(args.workdir, name + ".exp")
        if not os.path.exists(inp):
            t0 = time.time()
            sh([gen, inp, str(N), str(E), str(S), str(delta), str(K), typ, str(wmin), str(wmax), str(seed)])
            print(f"  generated in {time.time() - t0:.1f}s", flush=True)
        if not os.path.exists(exp):
            t0 = time.time()
            sh([ref, inp, exp])
            print(f"  reference in {time.time() - t0:.1f}s", flush=True)

        outs, times, status = [], [], "PASS"
        for r in range(args.runs):
            out = os.path.join(args.workdir, f"{name}.run{r}.out")
            t0 = time.time()
            try:
                p = subprocess.run([sol, inp, out], capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"  run {r}: {RED}TIMEOUT{NC} (> {args.timeout}s)")
                status = "TIMEOUT"
                break
            dt = time.time() - t0
            times.append(dt)
            if p.returncode != 0:
                print(f"  run {r}: {RED}CRASH{NC} exit={p.returncode}\n  stderr: {p.stderr.strip()[:500]}")
                status = "CRASH"
                break
            if p.stderr.strip():
                print(f"  run {r} stderr: {p.stderr.strip()[:500]}")
            outs.append(out)
            if same(out, exp):
                print(f"  run {r}: {GREEN}OK{NC} {dt:.2f}s", flush=True)
            else:
                cnt, diffs = first_diff(out, exp)
                print(f"  run {r}: {RED}WRONG{NC} {dt:.2f}s, {cnt} differing lines; first ones (line: got | expected):")
                for ln, got, want in diffs:
                    print(f"    {ln}: {got} | {want}")
                status = "WRONG"

        # determinism across runs (matters even when all are wrong)
        nondet = len(outs) > 1 and any(not same(outs[0], o) for o in outs[1:])
        if nondet:
            print(f"  {YELLOW}NON-DETERMINISTIC{NC}: runs produced different outputs")
            if status == "PASS":
                status = "WRONG"
        results.append((name, status, times, nondet))

        # keep disk usage down on Colab: drop outputs of passing runs
        if status == "PASS":
            for o in outs:
                os.remove(o)

    print("\n" + "=" * 64)
    print(f"{'test':<18}{'result':<10}{'min s':>8}{'max s':>8}  notes")
    for name, status, times, nondet in results:
        color = GREEN if status == "PASS" else RED
        tmin = f"{min(times):.2f}" if times else "-"
        tmax = f"{max(times):.2f}" if times else "-"
        note = "non-deterministic" if nondet else ""
        print(f"{name:<18}{color}{status:<10}{NC}{tmin:>8}{tmax:>8}  {note}")
    passed = sum(s == "PASS" for _, s, _, _ in results)
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    main()
