# Profiling findings — `multi_reflections_timed.cu`

Status as of 2026-10-29. Covers benchmarking, nsys, and ncu (all sections + roofline).
The ncu **Source** page has not been analysed yet (see *Open items*).

---

## 0. Setup

| Item | Value |
|---|---|
| GPU | Tesla T4 (Turing, CC 7.5), 40 SMs, Google Colab |
| CUDA / driver | nvcc 13.0, driver 580.82.07 |
| Tools | Nsight Systems 2025.3.1, Nsight Compute 2025.3.1 (viewed in ncu-ui 2026.3.1) |
| Image | 800 × 600, block 32×32 (1024 threads), grid 25×19 = 475 blocks |
| Profiled config | N = 400 spheres (+ floor), depth 10, threshold 0.01 |

**Scripts**
- `run.sh [depth] [threshold]` — builds once, one untimed warm-up run, then sweeps
  N ∈ {10, 20, 50, 100, 200, 300, 400}, 5 runs each, prints mean/std/min/max kernel time
  (timed with `cudaEvent` around the kernel launch only).
- `profile.sh [N] [depth] [threshold]` — builds with `-lineinfo`, then:
  1. `nsys profile --trace=cuda --stats=true` → `nsys_report.nsys-rep`, `nsys_stats.txt`
  2. `ncu --set full --import-source yes` → `ncu_report.ncu-rep`; key sections printed to `ncu_details.txt`
  3. `ncu --metrics <FLOP + byte counters>` → `ncu_roofline.txt`

**Caveat on times:** ncu locks the SM clock to base (585 MHz vs ~1590 MHz boost), so ncu
durations (~10.7 ms) are ~2.6× the `run.sh` numbers (~4.1 ms). Compare percentages/ratios
across tools, not milliseconds. The program's own "Kernel time" printed under ncu is
meaningless (ncu replays the kernel inside the event window).

---

## 1. Benchmark sweep (`run.sh`, depth 10, threshold 0.01)

| N | mean (ms) | std | min | max |
|---|---|---|---|---|
| 10 | 0.251 | 0.024 | 0.229 | 0.281 |
| 20 | 0.323 | 0.017 | 0.311 | 0.356 |
| 50 | 0.559 | 0.003 | 0.556 | 0.564 |
| 100 | 0.991 | 0.006 | 0.987 | 1.002 |
| 200 | 1.955 | 0.022 | 1.922 | 1.986 |
| 300 | 2.958 | 0.036 | 2.930 | 3.025 |
| 400 | 4.106 | 0.018 | 4.084 | 4.124 |

- **Linear in N, ~0.01 ms per sphere.** Expected: every ray (primary, each bounce, each
  shadow ray) loops over all N spheres in `closestHit`.
- **Warm-up matters.** Without it, the very first run of the session took 2.37 ms at N=10
  (others ~0.23 ms) — GPU coming out of idle / clock ramp. One untimed run before the sweep fixed it.
  It was *not* per-process lazy module loading (that would have inflated every run).

---

## 2. Nsight Systems — where does wall time go?

*How:* `nsys_stats.txt` (tables `cuda_api_sum`, `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`), and the timeline in `nsys-ui`.

| What | Time | Note |
|---|---|---|
| First `cudaMalloc` | 179.6 ms | CUDA context creation (lazy init), one-time — not allocation cost |
| `renderKernel` | 9.74 ms | No warm-up in `profile.sh`, so slower than `run.sh` |
| Image copy D→H (1.44 MB) | 0.114 ms | |
| Scene copy H→D (14 KB) | 0.003 ms | |

**Finding:** the kernel is ~99% of GPU time. Host↔device transfers are negligible →
all optimization effort belongs inside `renderKernel`. No concurrency/overlap to analyse
(single stream, one kernel), so the timeline adds nothing beyond the stats here.

---

## 3. Nsight Compute — kernel analysis

*How:* `ncu_details.txt` (text), plus `ncu-ui` → Page **Details** for charts.

### 3.1 Speed of Light (% of hardware peak)

| Metric | Value |
|---|---|
| Compute (SM) Throughput | 66.11 % |
| Memory Throughput | 66.11 % |
| L1/TEX Cache Throughput | 72.40 % |
| L2 Cache Throughput | 0.04 % |
| DRAM Throughput | 0.06 % |
| Duration (at 585 MHz) | 10.75 ms |

- Neither side saturated (>80%) nor idle (<40%) → mixed picture.
- **DRAM and L2 are idle.** The scene (400 × 36 B ≈ 14 KB) lives in L1 after first touch.
- Memory 66.11% vs L1 72.40% is the same activity measured over different windows:
  72.40 × (SM active cycles 5,738,407 / elapsed cycles 6,287,442) ≈ 66.1.

### 3.2 Memory Workload Analysis

| Metric | Value | Meaning |
|---|---|---|
| Local memory spilling | 0 | 44 regs/thread, no spills |
| DRAM bandwidth used | 193 MB/s | vs ~320 GB/s peak |
| L1 hit rate | 99.92 % | scene served from L1 |
| L2 hit rate | 78.8 % | of a tiny amount of traffic — irrelevant |
| Mem Busy (caches) | 33 % | |
| **Mem Pipes Busy (LSU)** | **66 %** | |

**Finding:** caches have spare bandwidth; the busy part is the load/store *instruction* side.

### 3.3 Compute workload, scheduler, warp state

| Metric | Value | Meaning |
|---|---|---|
| Executed IPC (active) | 2.66 / 4 | T4 SM = 4 schedulers × 1 issue/cycle |
| Busiest *math* pipe | FMA 56.5 % | |
| Active warps / scheduler | 6.55 / 8 | |
| Eligible warps / scheduler | 2.41 | |
| No eligible (wasted cycles) | 29.7 % | |
| Warp cycles per issued instruction | 9.32 | |
| Active threads per warp | 21.67 / 32 | divergence — ncu est. 22.6 % speedup if removed |

Consistency check: 6.55 warps ÷ 9.32 cycles ≈ 0.70 issues/cycle/scheduler = reported value.

### 3.4 Stall reasons (cycles per issued instruction; `ncu-ui` → Warp State Statistics chart)

| Reason | Cycles | Meaning |
|---|---|---|
| Stall Wait | 2.48 | fixed-latency dependency on own previous (math) result |
| Stall Not Selected | 2.43 | ready, but scheduler picked another warp (not wasted for the SM) |
| Stall Math Pipe Throttle | 1.55 | ready, but FP32 pipe busy (16 lanes/scheduler → 2 cycles per warp FP32 inst) |
| Selected | 1.00 | the issue cycle itself |
| Stall LG Throttle | 0.54 | global/local load queue full (load instruction pressure) |
| Stall Branch Resolving | 0.51 | branches / divergence |
| Stall Long Scoreboard | 0.34 | waiting for memory data — small (L1 hits) |
| Stall No Instruction | 0.33 | instruction fetch |
| rest | ~0.02 | |

Bars sum to ≈ 9.32 (sanity check). **Finding:** waiting is dominated by arithmetic
(dependency chains + FP32 pipe), not memory latency.

### 3.5 Launch stats & occupancy

| Metric | Value |
|---|---|
| Registers/thread | 44 (65,536 / 1024 = 64 max for this block size — launch fails above) |
| Block limit — registers | 1 block/SM (44 × 1024 = 45,056 regs) |
| Block limit — warps | 1 block/SM (32 warps = SM max) |
| Theoretical occupancy | 100 % (32/32 warps) |
| **Achieved occupancy** | **77.6 %** (24.8 warps) — ncu est. 22.4 % speedup |
| Waves per SM | 11.88 |

**Finding:** with one 1024-thread block per SM, a new block can only start when *all*
32 warps finish. Warps differ hugely in work (sky pixels: 1 iteration; reflective pixels:
up to 11 bounces + shadow rays), so finished warps' slots sit idle. Minor extras: partial
last wave (35/40 SMs), bottom block row has 8 dead warps (600 = 18.75 × 32 rows).

### 3.6 Source page — where the stalls are (`ncu-ui` → Source, Left = Source, Right = SASS, Navigate By = Warp Stall Sampling)

Share of warp stall samples per source line (inlined helpers are attributed to their own lines):

| Line | Code | Stall samples |
|---|---|---|
| 50 | `Vec3 oc = origin - s.center;` | **23.71 %** |
| 29 | `dot(...)` (inlined helper) | 12.45 % |
| 27 | `operator-` (inlined helper) | 9.69 % |
| 54 | `float disc = b*b - 4*a*c;` | 9.46 % |
| 55 | `if (disc < 0) return -1.0f;` | 7.63 % |
| 52 | `float b = 2.0f * dot(oc, dir);` | 6.82 % |
| 64 | `for (int i = 0; i < scene_size; i++)` | 6.65 % |
| 65 | `float t = hitSphere(scene[i], ...)` | 5.75 % |
| 53 | `float c = dot(oc, oc) - s.radius*s.radius;` | 4.52 % |
| 66 | `if (shadow && t > 0) return i;` | 3.47 % |
| 57 | `return (t > 0.001f) ? t : -1.0f;` | 3.11 % |
| 67 | `if (t > 0 && t < closest) ...` | 2.83 % |
| 56 | `float t = (-b - sqrt(disc)) / (2*a);` | 1.72 % |
| 51 | `float a = dot(dir, dir);` | no samples visible (likely hoisted out of the loop — loop-invariant) |

- **~98 % of all stall samples are in the sphere loop** (`hitSphere` + its inlined helpers + `closestHit`).
  Shading, reflection maths, and pixel write in `rayColor`/`renderKernel` are negligible.
- **SASS for line 50:** `IMAD.WIDE` (address = `scene + i * 36`) followed by **4 separate
  `LDG.E.SYS`** — scalar 32-bit global loads for `center.x`, `center.y`, `center.z`, `radius`.
  Confirms the hypothesis: the 36-byte, 4-byte-aligned `Sphere` prevents a single wide
  (e.g. 128-bit) load, so each sphere test issues 4 load instructions — the LSU/L1-request
  pressure seen in §3.2 and §4.3. The loads are plain global loads (not the read-only/constant path).
- Line 50 is also where those loaded values are first *consumed*, so waits for the loads land there.
- `dot` SASS = `FMUL` → `FFMA` → `FADD`: a 3-instruction dependent chain (each needs the previous
  result) — the source of the "Stall Wait" in §3.4. Line 54 (`b*b - 4*a*c`) is another chain.
- Line 55 (`disc < 0` branch, 7.6 %) is where lanes in a warp diverge (some hit, most miss).
- **Stall split for line 50** (hover on its bar; 69,962 samples):

  | Reason | Line 50 | Whole kernel (§3.4, ≈ share of 9.32 cycles) |
  |---|---|---|
  | Wait | 23.5 % | ~27 % |
  | Not Selected | 18.4 % | ~26 % |
  | **Long Scoreboard** | **15.2 %** | ~4 % |
  | **LG Throttle** | **10.9 %** | ~6 % |
  | Math Pipe Throttle | 10.8 % | ~17 % |
  | Selected | 8.9 % | ~11 % |
  | Branch Resolving | 8.1 % | ~5 % |

  Memory-related stalls (Long Scoreboard + LG Throttle) are ~26 % on line 50 vs ~10 %
  kernel-wide → the sphere loads' cost concentrates here. By sample count, line 50 holds
  roughly all of the kernel's Long Scoreboard stalls. Kernel-wide, memory waiting is still
  a minority — arithmetic dependency/pipe stalls dominate.
- **Warning triangle hint** (lines 50/53/54): "This line executes many non-fused floating-point
  instructions … consider converting pairs of non-fused instructions to FMA … enable
  `--use_fast_math` or `--fmad=true`." Note: `--fmad=true` is already nvcc's default, so the
  remaining FMUL/FADD pairs are ones the compiler couldn't fuse as written (expression order).
  `--use_fast_math` additionally uses approximate `sqrt`/division and flushes denormals —
  a precision trade-off, so the image must be compared after enabling it.

---

## 4. Roofline

### 4.1 Hand calculation (`ncu_roofline.txt`)

| Quantity | Source | Value |
|---|---|---|
| fadd / fmul / ffma (thread-level) | `sm__sass_thread_inst_executed_op_{fadd,fmul,ffma}_pred_on.sum` | 1.824 G / 1.380 G / 2.755 G |
| FP64 ops | `..._op_{dadd,dmul,dfma}_pred_on.sum` | 0 (no accidental doubles) |
| FLOPs | fadd + fmul + 2·ffma | **8.71 GFLOP** |
| DRAM bytes | `dram__bytes_{read,write}.sum` | 1.22 + 1.02 = 2.24 MB |
| Time | `gpu__time_duration.sum` | 10.74 ms |
| Achieved | 8.71 G / 10.74 ms | **811 GFLOP/s** |
| Peak FP32 @585 MHz | 40 SM × 64 FP32 × 2 × 0.585 GHz (Turing spec) | 2,995 GFLOP/s (8,141 @ boost) |
| Peak DRAM BW | T4 spec | ~320 GB/s |
| Ridge point | 2,995 / 320 | 9.4 FLOP/B |
| AI (DRAM) | 8.71 G / 2.24 M | **≈ 3,900 FLOP/B** |

Assumption: ncu locked only the SM clock, not memory (`DRAM Frequency 5.00 GHz` supports this).
Conclusion is insensitive to it.

### 4.2 ncu-ui charts (Speed of Light section → chart dropdown)

- **GPU Throughput Rooflines** (DRAM only): dot at AI ≈ 3,900, ≈ 0.8 TFLOP/s, far right of
  the 9.4 ridge, under the ~3 TFLOP/s FP32 roof → **compute-bound w.r.t. DRAM, at 27 % of
  FP32 peak** (≈ 3.7× headroom).
- **Roofline Single Precision** (hierarchical): three bandwidth roofs with ridges at
  ≈1 (L1, ~3 TB/s), ≈3.5 (L2, ~0.9 TB/s), ≈9.4 (DRAM). Three dots at the same height:
  L1 at AI ≈ 0.8 (left of L1 ridge, ~⅓ of the L1 roof); the other two at ≈4,000 and
  ≈9,000 (DRAM and L2 — colour mapping not yet confirmed).

### 4.3 The trap — bytes vs requests (GPU Throughput Breakdown)

| Metric | % |
|---|---|
| L1: Lsuin Requests | **66.11** ← this *is* the SOL "Memory 66.11 %" |
| L1: Data Pipe LSU Wavefronts | 33.08 |
| L1: Data Bank Reads | 0.53 |
| SM: Inst Executed Pipe LSU | **66.11** ← this *is* the SOL "Compute 66.11 %" |
| SM: Pipe FMA Cycles Active | 56.55 |
| SM: Pipe ALU Cycles Active | 27.81 |
| SM: Inst Executed Pipe CBU (branches) | 25.45 |

- All 32 lanes of a warp read the **same** sphere field → L1 broadcasts one value; each
  request moves almost no data (bank reads 0.5 %) but still costs a request slot.
- A byte-based roofline cannot see this: by bytes L1 looks ~⅓ used; by request rate it's 66 %.
- **Both SOL headline numbers come from load/store instructions**, not FP math.
  (Earlier in the analysis I attributed the SM 66 % to arithmetic — that was wrong.)

---

## 5. Summary of bottlenecks

1. **Busiest unit: load/store instruction issue** (LSU pipe / L1 requests, 66 %) — from
   re-reading sphere data for every ray × sphere × bounce × shadow ray. Not a bandwidth or
   latency problem (L1 hit 99.9 %, long-scoreboard stalls small).
2. **Arithmetic stalls dominate warp waiting** — dependency chains (Stall Wait 2.48) and
   FP32 pipe pressure (Throttle 1.55).
3. **Divergence:** ~21.7 of 32 lanes active (ncu est. 22.6 %).
4. **Intra-block load imbalance:** achieved occupancy 77.6 % vs 100 % theoretical
   (ncu est. 22.4 %), amplified by 1 block/SM.
5. **Not bottlenecks:** DRAM, L2, host↔device transfers, register spilling, FP64.
6. Overall: compute-bound per roofline, at 27 % of FP32 peak.

## 6. Candidate directions (hypotheses, not yet tested)

To discuss — each should be measured with `run.sh` and re-profiled with `profile.sh`:
- **Fewer/cheaper scene loads:** `Sphere` layout (struct is 36 B, not 16-B aligned →
  confirmed 4 scalar loads per sphere test, §3.6), read-only load path, shared memory staging
  (on Turing, shared mem and L1 are the same SRAM — expected gain modest, since data is
  already L1-resident), `__constant__` memory (built for warp-uniform broadcast reads).
- **Divergence / imbalance:** block shape and size (smaller blocks → several blocks per SM
  so finished warps can be backfilled).
- **Arithmetic:** shortening dependency chains / instruction count in `hitSphere`; more FMA
  fusion (ncu hint); `--use_fast_math` as a build-flag experiment (check image quality).
- **Algorithmic:** testing fewer spheres per ray (e.g. spatial culling/BVH) — attacks
  the linear-in-N cost directly.

## 7. Open items

- [x] ncu **Source** page: stalls are ~98 % in the sphere loop; 4 scalar `LDG` per sphere (§3.6).
- [x] Warning triangles and line-50 stall breakdown (§3.6).
- [ ] Hierarchical roofline: hover the dots to confirm which colour is L2 vs DRAM, and the
      L1 byte count ncu uses (it implies ~4× more than `l1tex__t_bytes` = 2.67 GB).
- [ ] Re-profile with the warm-up in place if nsys timings are needed for the report.
