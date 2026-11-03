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

**Report locations** (relative to the project folder): baseline (`multi_reflections_timed.cu`)
reports are in `profiles/`, shared-memory version reports in `profiles_shared/`. Report file names
below (`ncu_details.txt`, `ncu_roofline.txt`, …) refer to files inside those folders.

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

## 7. Optimization 1 — scene copied to shared memory (`multi_reflections_shared.cu`)

*Change:* at kernel start, each block cooperatively copies `scene[0..N]` into dynamic shared
memory, `__syncthreads()`, then all ray tests read the shared copy (bounds-check moved below the
barrier). Launch passes `N+1 × 36 B` of dynamic shared memory; opt-in via `cudaFuncSetAttribute`
above 48 KB; hard limit 64 KB/block on T4 (~1,820 spheres). Output verified pixel-identical to the
baseline with `compare_images.py` (N=400, depth 10, threshold 0.01).

*Timing* (`run.sh` vs `SRC=multi_reflections_shared.cu bash run.sh`, depth 10, threshold 0.01, 5 runs;
raw output with std/min/max in `results/run_baseline.txt` and `results/run_shared.txt`):

| N | baseline (ms) | shared (ms) | speedup |
|---|---|---|---|
| 10 | 0.266 | 0.149 | 1.79× |
| 20 | 0.317 | 0.207 | 1.53× |
| 50 | 0.563 | 0.433 | 1.30× |
| 100 | 0.991 | 0.830 | 1.19× |
| 200 | 1.937 | 1.701 | 1.14× |
| 300 | 2.960 | 2.640 | 1.12× |
| 400 | 4.105 | 3.717 | 1.10× |

- Consistent gain at every N; ~10 % at N=400, matching the expectation that shared memory
  helps only modestly when the scene is already L1-resident (still 4 load instructions per sphere).
- Per-sphere slope (N 100→400): 0.0104 → 0.0096 ms/sphere (~7 % lower).
- Larger relative gain at small N (1.8× at N=10) — not yet explained; hypothesis: the baseline's
  first-touch global loads are exposed when there is little work to hide them. To check with a profile.
*Profile at N=400* (`SRC=multi_reflections_shared.cu OUT=profiles_shared bash profile.sh 400 10 0.01`,
reports in `profiles_shared/`; both at ncu's locked 585 MHz):

| Metric | Baseline | Shared |
|---|---|---|
| Duration | 10.75 ms | 10.02 ms (−6.8 %) |
| Elapsed cycles | 6.29 M | 5.86 M |
| Memory Throughput (SOL) | 66.1 % | 45.4 % |
| L1/TEX Throughput | 72.4 % | 52.7 % |
| Mem Pipes Busy | 66.1 % | 45.4 % |
| Compute (SM) Throughput | 66.1 % | 61.0 % |
| FMA pipe | 56.5 % | 60.7 % (now the busiest pipe) |
| `l1tex__t_bytes` | 2.67 GB | 59 MB |
| FLOPs (fadd/fmul/ffma) | identical | identical |
| IPC / warp cycles per issued inst | 2.66 / 9.32 | 2.67 / 9.33 |
| Active threads per warp | 21.7 | 21.5 |
| Achieved occupancy | 77.6 % | 77.8 % |
| L1 hit rate | 99.9 % | 96.6 % (fewer global loads, so first-touch misses are a larger share) |
| Block limit (shared mem) | 16 | 2 (irrelevant: registers/warps already limit to 1 block/SM) |

- The load-issue bottleneck (66 %) dropped to 45 %; ncu's verdict changed from "compute and
  memory well-balanced" to "compute more heavily utilized than memory" — the FMA pipe is now the top unit.
- Same IPC but ~7 % fewer cycles ⇒ ~7 % fewer instructions executed. Likely cause: shared-memory
  addressing is 32-bit, removing 64-bit global address arithmetic (`IMAD.WIDE`) — to confirm in Source view.
- Divergence and occupancy/imbalance unchanged (not targeted) — those ~22 % ncu estimates remain.
- **Source view (shared version, `oc = origin - s.center` is line 51 there):** SASS shows
  **`LDS.U.128`** — one 128-bit shared load fetching `center.xyz + radius` — instead of the baseline's
  4 × 32-bit `LDG.E.SYS`. The compiler knows the alignment of the `extern __shared__` array and can
  use wide loads where `s_scene[i]` is 16-B aligned (every 4th sphere with a 36-B stride); through the
  global `Sphere*` it only knows 4-B alignment, so it must use scalar loads. This explains the ~7 %
  fewer instructions. Line 51's instruction mix: 39 instructions (23 FP, 13 load/store, 3 integer),
  spanning both inlined copies of the loop — exact loads per sphere not yet counted.
- Stall samples shifted from the load line to the next consumer: `oc = …` 23.7 % → 16.9 %,
  `b = 2*dot(oc,dir)` 6.8 % → 15.8 %; `hitSphere` total unchanged (~55 %).
- **Implication for data layout (§6):** a 16-B-aligned `Sphere` (padding to 48 B) or a separate
  `float4` array for `center+radius` would let *every* sphere test use one 128-bit load — from
  global memory as well as shared.
- **Stall split on the `oc = origin - s.center` line** (hover on its stall bar; samples ∝ time):

  | Stall | Baseline (line 50) | Shared (line 51) |
  |---|---|---|
  | Long Scoreboard (L1/global load wait) | 10,623 (15.2 %) | 0 |
  | LG Throttle (global load queue full) | 7,655 (10.9 %) | 0 |
  | Short Scoreboard (shared-mem load wait) | 0 | 9,403 (19.6 %) |
  | MIO Throttle (shared-mem instruction queue) | 0 | 971 (2.0 %) |
  | Wait | 16,453 | 7,474 |
  | Not Selected | 12,852 | 10,481 |
  | Math Pipe Throttle | 7,546 | 6,242 |
  | Branch Resolving | 5,660 | 5,807 |
  | **Total samples** | **69,962** | **47,948 (−31 %)** |

  Global-memory stalls disappear and are replaced by their shared-memory counterparts; memory-related
  waiting on this line drops ~43 % (18,278 → 10,374 samples): fewer instructions (128-bit load) and
  lower latency. Branch stalls unchanged (divergence not targeted).

*Next (proposed for next week):* tiling for scenes larger than shared memory (see discussion:
L1 ≈ 1,800 spheres, L2 ≈ 116k; requires lock-step bounce loop so all threads reach `__syncthreads()`).

## 8. Larger scenes — L1 capacity (`compare_sizes.sh`, scene extended to 5,000 spheres)

*How:* `bash compare_sizes.sh` (depth 10, threshold 0.01, 3 runs per N); raw output in
`results/compare_sizes_shared.txt`. "ms/sph" = marginal cost of each extra sphere since the previous row.

| N | baseline (ms) | shared (ms) | speedup | base ms/sph | shared ms/sph |
|---|---|---|---|---|---|
| 400 | 4.100 | 3.708 | 1.11× | – | – |
| 600 | 6.073 | 5.537 | 1.10× | 0.00987 | 0.00914 |
| 800 | 8.003 | 7.356 | 1.09× | 0.00965 | 0.00909 |
| 1000 | 9.979 | 9.197 | 1.09× | 0.00988 | 0.00920 |
| 1200 | 12.026 | 11.119 | 1.08× | 0.01024 | 0.00961 |
| 1364 | 13.663 | 12.609 | 1.08× | 0.00998 | 0.00909 |
| 1365 | 13.673 | 12.630 | 1.08× | – | – |
| 1500 | 15.500 | 13.912 | 1.11× | 0.01353 | 0.00950 |
| 1700 | 19.333 | 15.832 | 1.22× | 0.01916 | 0.00960 |
| 1819 | 24.353 | 17.018 | 1.43× | 0.04218 | 0.00997 |
| 1820 | 24.264 | fail | – | – | – |
| 2000 | 28.582 | fail | – | 0.02399 | – |
| 2500 | 36.428 | fail | – | 0.01569 | – |
| 3000 | 44.852 | fail | – | 0.01685 | – |
| 4000 | 61.510 | fail | – | 0.01666 | – |
| 5000 | 78.839 | fail | – | 0.01733 | – |

- **Repeat run** (`results/compare_sizes_shared.txt`, the latest run): every number agrees with the table
  above to within ~2 % (5000: 80.541 vs 78.839 ms), except baseline N=400 (4.402 vs 4.100 ms,
  so a speedup of 1.18× vs 1.11×). That point comes first in the sweep, so it is probably GPU warm-up
  noise. Treat the N=400 speedup as ~1.1–1.2×. The cliff (1.44× at N=1819) and the ~1.7× L2
  plateau reproduce.
- **Shared version:** flat ~0.0091–0.0100 ms/sphere up to its limit. Works above 48 KB
  (opt-in via `cudaFuncSetAttribute` OK at N=1365); fails cleanly at N=1820 (64 KB limit) as designed.
- **Baseline:** flat ~0.0099 ms/sphere up to N≈1,364, then degrades: 1.4× (→1500), 1.9× (→1700),
  **4.3× (1700→1819, thrashing)**, then settles at **~1.7×** (N ≥ 2,500) — the L2-resident regime
  (L2 = 4 MB ≈ 116k spheres, so this level should hold far beyond 5,000).
- The cliff starts at ~50–54 KB of scene, earlier than the nominal 64 KB L1: a cyclic scan close to
  cache capacity already evicts lines just before reuse (LRU-like replacement, conflicts).
- Shared memory's advantage grows near capacity (1.08× → 1.43× at N=1819): explicitly managed
  storage is never evicted. Strong motivation for **tiling** (next week) to keep the flat
  ~0.0095 ms/sphere beyond 1,820 spheres.
- Profiles planned: baseline N=1819 (`OUT=profiles_1819`), shared N=1819 (`OUT=profiles_shared_1819`),
  baseline N=3000 (`OUT=profiles_3000`). Expected: baseline L1 hit rate falls, L2 traffic and
  Long Scoreboard stalls rise; shared unchanged vs N=400.

## 9. Profile at N=5000 — baseline in the L2-resident regime

*How:* `OUT=profiles_5000 bash profile.sh 5000` (depth 10, threshold 0.01), reports in `profiles_5000/`.
Shared version not profiled: it cannot hold 5,000 spheres. Stall breakdown from
`ncu --import <rep> --page raw | grep "warps_issue_stalled_.*_per_issue_active"`.
Both columns at ncu's locked 585 MHz.

| Metric | N=400 | N=5000 |
|---|---|---|
| Duration | 10.75 ms | 184.34 ms (17.1× for 12.5× spheres → 1.37× per sphere) |
| Compute (SM) / L1 SoL | 66.1 % / 72.4 % | 51.5 % / 55.5 % |
| L1 hit rate | 99.92 % | 98.56 % |
| L1 bytes (`l1tex__t_bytes`) | 2.67 GB | 35.83 GB |
| L2 bytes (`lts__t_bytes`) | 3.51 MB | 518.6 MB (148×) |
| L2 SoL / DRAM bytes | 0.04 % / 2.2 MB | 0.32 % / 2.9 MB |
| No eligible warp | 29.7 % | 45.5 % |
| Eligible warps / scheduler | 2.41 | 1.71 |
| Cycles per issued instruction | 9.32 | 11.43 |
| Achieved occupancy | 77.6 % | 72.1 % |
| Active threads / warp | 21.7 | 20.4 |
| FMA pipe utilisation | 56.5 % | 43.6 % |

*Stall breakdown* (cycles per issued instruction; sums to the 9.32 / 11.43 above):

| Reason | N=400 | N=5000 | Δ |
|---|---|---|---|
| wait | 2.48 | 2.48 | 0 |
| not_selected | 2.43 | 2.13 | −0.30 |
| **long_scoreboard** | 0.34 | **2.14** | **+1.80** |
| math_pipe_throttle | 1.55 | 1.43 | −0.12 |
| **lg_throttle** | 0.54 | **1.43** | **+0.89** |
| selected | 1 | 1 | 0 |
| branch_resolving | 0.51 | 0.50 | 0 |
| no_instruction | 0.33 | 0.23 | −0.10 |
| dispatch_stall | 0.12 | 0.08 | −0.04 |

- **Latency-bound, not bandwidth-bound.** Every unit is *less* busy than at N=400; L2 bandwidth is
  essentially idle (0.32 %) and DRAM is untouched. The slowdown is waiting, not throughput.
- **The whole slowdown is memory stalls.** long_scoreboard (+1.80, warps waiting on L1 misses served
  by L2) and lg_throttle (+0.89, LSU queue for global loads full) add +2.69 cycles/instruction; the
  rest shrank slightly. Net +2.11 = 11.43 − 9.32.
- **Why the L1 hit rate still looks high.** Reuse *within one scan* survives: all 32 threads of a
  warp read the same sphere (one request), and the warps of an SM walk the list at roughly the same
  pace, so the first warp misses and the following ones hit. What capacity evictions kill is reuse
  *between scans* (next bounce / shadow ray restarting at sphere 0). One miss per line per scan vs
  dozens of hits, hence 1.44 % misses, yet 18× the miss rate and 148× the L2 traffic of N=400.
- **lg_throttle points at load count, not bytes.** 4 scalar `LDG` per sphere (§3.6) fill the LSU
  queue faster while misses are outstanding. One 128-bit load would issue 4× fewer instructions.
- **Occupancy is limited by block granularity, not by too many warps.** 44 regs × 1,024 threads
  ⇒ exactly 1 block/SM (Block Limit Registers = 1). Warps whose rays finish early leave their slots
  idle until the whole block retires ⇒ 72 % achieved vs 100 % theoretical (ncu est. ≤ 28 %).
- **Compute is not the limit.** FMA pipe 43.6 %; math_pipe_throttle unchanged (~1.4–1.5).
- nsys kernel time with the warm-up: 103.0 ms vs 80.5 ms in `compare_sizes.sh`. The warm-up closed most
  of the gap seen at N=400 (9.74 vs 4.1 ms) but not all of it. Quote `compare_sizes.sh` times.

*Implications for the planned optimizations:*

| Idea | Verdict at N=5000 |
|---|---|
| Fewer warps per SM | Opposite: need *more* eligible warps. Try smaller blocks (e.g. 16×16) so several blocks share an SM and finished ones are replaced promptly. |
| FMA / arithmetic tweaks | Lowest priority: FMA pipe < 50 %, math throttle unchanged. |
| Read less / selectively | Strongest lever. Intersection needs only center + radius (16 B of 36 B): a separate float4 array fits ~2.25× more spheres in L1 (cliff ~1,800 → ~4,000) and turns 4 `LDG` into 1 (cuts lg_throttle). Tiling attacks the same misses. |
| Divergence (new) | 20.4 / 32 active threads (ncu est. ≈ 20 %); harder to fix. |

## 10. Optimization 2 — hot/cold split, center + radius as `float4` (`multi_reflections_split.cu`)

*Change:* the sphere loop (`closestHit` / `hitSphere`) reads a separate `float4` array
`(cx, cy, cz, r)`, 16 B and 16-B aligned, built on the host with `make_float4`. The full
`Sphere` array (36 B) is still used only for the hit sphere (normal, color, floor, reflectivity).
Motivated by §9: lg_throttle (4 scalar `LDG` per sphere) and L1 misses.

*Profile at N=5000* (`SRC=multi_reflections_split.cu OUT=profiles_split_5000 bash profile.sh 5000`,
locked 585 MHz):

| Metric | Baseline | Split | Change |
|---|---|---|---|
| Duration | 184.3 ms | 154.7 ms | **1.19× faster** |
| FP32 fadd / fmul / ffma | 22.96 / 17.23 / 34.46 G | identical | same work |
| L1 bytes | 35.83 GB | 8.97 GB | 4.0× less |
| L2 bytes | 518.6 MB | 211.5 MB | 2.45× less (36 → 16 B predicts 2.25×) |
| Mem Pipes Busy | 51.5 % | 15.5 % | |
| L1 hit rate | 98.56 % | 97.67 % | lower: see below |
| SoL SM / memory | 51.5 / 51.5 % | 52.3 / 24.0 % | |
| SM active cycles | 100.9 M | 84.4 M | −16 % |
| Instructions (IPC × active cycles) | ~204 M / SM | ~182 M / SM | ~−10 % |
| Cycles per issued instruction | 11.43 | 11.04 | |
| Eligible warps / scheduler | 1.71 | 1.88 | |
| Achieved occupancy | 72.1 % | 74.6 % | |

*Stall breakdown* (cycles per issued instruction; "abs." = ratio × instructions, ∝ total stall cycles):

| Reason | Baseline | Split | abs. change |
|---|---|---|---|
| **lg_throttle** | 1.43 | **0.01** | **≈ −100 %** |
| long_scoreboard | 2.14 | 2.67 | +12 % |
| math_pipe_throttle | 1.43 | 1.77 | +11 % |
| not_selected | 2.13 | 2.24 | −6 % |
| wait | 2.48 | 2.41 | −13 % |
| branch_resolving | 0.50 | 0.57 | +2 % |
| no_instruction | 0.23 | 0.29 | |
| **total** | 11.43 | 11.04 | **−14 %** (matches −16 % cycles) |

- **lg_throttle eliminated** as predicted: one `LDG.E.128` per sphere instead of four scalar loads.
- **Lower L1 hit rate is a denominator artifact.** In the baseline, loads 2–4 of a sphere hit the
  sector that load 1 just brought in ("free" hits). The split removes those requests, so the rate
  falls while real misses (L2 bytes) drop 2.45×.
- **Removing a throttle exposes the latency under it.** The waiting moved from "cannot issue the
  load" (lg_throttle) to "waiting for its data" (long_scoreboard, +12 % absolute). The scene is
  still past L1 at N=5000: 5,001 × 16 B = 80 KB > 64 KB, so the split is past its own cliff
  (predicted ~3,200–3,400). long_scoreboard should drop below that N.
- **math_pipe_throttle rose** (+11 %): with loads no longer gating issue, the FMA pipe is closer to
  its limit. FMA / arithmetic tweaks (idea 2) become somewhat more relevant than in §9.
- nsys single run: 111.8 ms split vs 103.0 ms baseline, *opposite* to ncu. The nsys/ncu ratio differs
  (0.72 vs 0.56), so these were probably different clocks (Colab T4 throttling, separate sessions).
  Not used. `compare_sizes.sh` (3 runs, back-to-back) is the reference.
- *ncu-ui SoL (split vs baseline):* SM +1.5 %, memory −53 %, L2 −51.5 %, **L1/TEX only −13 %**
  despite 4× fewer `l1tex__t_bytes` (bytes vs requests again, §4.3). Duration −16.1 %.
- *Roofline Single Precision (hierarchical), N=5000.* Dot colours: L1 purple, L2 orange,
  DRAM teal; outline green = baseline, blue = current. Ridges ≈ 1 (L1), ≈ 3.5 (L2), ≈ 9.4 (DRAM).

  | Level | Baseline AI | Split AI | AI from counters (base / split) |
  |---|---|---|---|
  | L1 (Global/Local) | 0.76 | 0.98 | 3.0 / 12.2 |
  | L2 | 211.38 | 523.05 | 210 / 516 (match within ~1 %) |
  | DRAM | 37,426 | 37,743 | |

  Exact values from the ncu-ui chart export. Performance 591.8 → 705.2 GFLOP/s; FP32 roof 2,999 GFLOP/s.
  Implied L1 bytes (109.1 GFLOP / AI): 143.6 GB baseline, 111.3 GB split.
  - **Every dot is ~4× below its roof** (L1 dots ~25 % of the L1 slope, all ~24 % of the FP32
    roof). No resource is saturated ⇒ **latency-bound**, consistent with the stalls.
  - **ncu's L1 bytes ≠ `l1tex__t_bytes`.** Factor 4.0× for the baseline but 12.4× for the split,
    so the L1 dot moved only 1.29× right instead of 4×. Hypothesis: the roofline counts bytes *requested
    by threads* (32 lanes × load width = 512 B per sphere per warp in *both* versions), while
    `l1tex__t_bytes` counts sectors (broadcast load ⇒ 4 × 32 B = 128 B baseline, 1 × 32 B split).
    That predicts 4× and 16×; observed 4.0× and 12.4×. Not yet verified (Raw page: `data_bytes` metrics).
- *Compute Workload, pipe utilization* (ncu-ui, % of peak over active cycles, ≈ read off the chart):

  | Pipe | Baseline | Split |
  |---|---|---|
  | FMA | ~47 % | ~56 % |
  | LSU | **~55 %** (top pipe) | ~16.5 % |
  | ALU | ~23 % | ~27.5 % |
  | CBU (branch) | ~5 % | ~6 % |
  | XU (sqrt, rcp) | ~0 | ~0 |

  - LSU down ~3.4×: 4 load instructions per sphere → 1. In the baseline LSU was the busiest pipe.
  - FMA and ALU rose only because the same instructions now fit in 16 % fewer cycles
    (47 / 0.84 ≈ 56). FMA is now the top pipe.
  - **XU ≈ 0:** `sqrt` and the divide in `hitSphere` run only when `disc ≥ 0` (rare), so they
    are not a target for arithmetic tweaks. Those should target the FMA-pipe work done for *every* sphere.
  - **Headroom:** FMA at 56 % ⇒ hiding all latency could give at most ~1.8× before the FMA pipe
    saturates. Beyond that, only fewer FP32 instructions per sphere help.
- *Scheduler:* 6.39 active vs 1.88 eligible warps per scheduler, 42 % of cycles with none eligible.
  ncu "Est. Speedup 42 %" = the No-Eligible % itself, i.e. an upper bound, not a prediction.
- *Warp State, levers per stall:* long_scoreboard (2.67, load latency) and wait (2.41, serial
  `oc → dot → disc` dependency chain) are 5.1 of 11.0 cycles. Both shrink with **more warps**
  (occupancy) or **more independent work per warp** (ILP: e.g. 2 spheres per loop iteration, so
  two loads and two independent chains overlap). math_pipe_throttle (1.77) needs fewer FP32
  instructions; not_selected (2.24) is harmless. Divergence: 20.36 / 32 active threads, unchanged
  from the baseline (mainly different bounce counts within a 32-pixel row, plus shadow early exit).
- *Occupancy:* theoretical 100 % already (32 warps/SM = one 1,024-thread block); achieved 74.6 %.
  The gap is block granularity / imbalance: a block's warps finish at different times, but the SM
  cannot start a new block until the whole block retires. ncu est. 25 % (upper bound).
  - Register graph: 100 % up to **64 regs/thread** (we use 44). Headroom for unrolling/ILP at no
    occupancy cost. Above 64, a 1,024-thread block cannot launch at all.
  - Block-size graph: 100 % theoretical at 64, 128, 256, 512, 1,024; dips in between (e.g. ~53 % just above 512).
  - Candidate: 256 threads (4 blocks/SM) or 128 (8 blocks/SM). Compare **32×8** (same warp shape
    as now, 32×1 pixel row: isolates the block-granularity effect) with **16×16** (warp = 16×2
    pixels, more compact, which may also change divergence).
- *Block-shape sweep* (`bash compare_blocks.sh`, split version, depth 10, threshold 0.01, 3 runs;
  block size = arguments 4–5 of `multi_reflections_split.cu`). Speedup vs 32×32 at the same N:

  | Block (threads, warp footprint) | N=400 | N=3000 | N=5000 |
  |---|---|---|---|
  | 32×32 (1024, 32×1) | 3.984 ms | 30.133 ms | 64.366 ms |
  | 32×16 (512, 32×1) | 1.08× | 1.08× | 1.12× |
  | 32×8 (256, 32×1) | 1.14× | 1.05× | 1.23× |
  | 32×4 (128, 32×1) | 1.18× | 1.12× | 1.26× |
  | 16×16 (256, 16×2) | 1.22× | 1.20× | 1.29× |
  | 16×8 (128, 16×2) | 1.27× | 1.23× | 1.38× |
  | **8×8 (64, 8×4)** | **1.30×** | **1.27×** | **1.39×** (46.2 ms) |

  All shapes render pixel-identical images to 32×32 (`compare_images.py`, N=400).
  Two separate effects, both monotonic:
  - **Block granularity** (same warp shape, smaller block): 32×32 → 32×4 gives 1.18–1.26×. Smaller
    blocks retire independently, so finished ones are replaced at once (achieved occupancy ↑).
    8×8 still allows 100 % theoretical: 16 blocks/SM × 2 warps = 32 warps.
  - **Warp footprint** (same block size, more compact warp): 32×8 → 16×16 and 32×4 → 16×8 add
    another ~5–10 %. A 16×2 or 8×4 pixel patch is more coherent than a 32×1 row: neighbouring rays
    hit the same objects and bounce the same number of times ⇒ less divergence.
  - Pixel locality does not affect scene caching (every ray scans all spheres), but block size does
    (see the 8×8 profile below: warps of small blocks scan out of step).
  - Noise: ±a few % (e.g. 32×8 at N=3000 below 32×16). 3 runs, Colab clocks.
  - Not to go smaller: a 32-thread block allows only 16 warps/SM (16-block limit) = 50 % theoretical.
- *Profile 8×8 vs 32×32* (split, N=5000, `profiles_split_8x8` vs `profiles_split_5000`, 585 MHz):

  | Metric | 32×32 | 8×8 |
  |---|---|---|
  | Duration | 154.7 ms | 112.6 ms (1.37×) |
  | Active threads / warp | 20.36 | **22.86** |
  | Instructions (IPC × active cycles) | ~182 M / SM | ~163 M / SM (−11 %) |
  | IPC (active) | 2.16 | **2.64** |
  | No eligible | 42.1 % | 31.6 % |
  | Issued warp / scheduler | 0.58 | 0.68 |
  | Eligible warps / scheduler | 1.88 | 1.81 |
  | Achieved occupancy | 74.6 % | 79.9 % (only +5 pts) |
  | Cycles per issued instruction | 11.04 | 9.73 |
  | SoL SM / memory | 52.3 / 24.0 % | 64.1 / 33.1 % |
  | FMA pipe | ~56 % | 64.1 % |
  | L1 hit rate | 97.67 % | 93.02 % |
  | L2 bytes | 211.5 MB | **560.7 MB (2.65×)** |

  - **1.37× = 1.12× fewer instructions × 1.22× higher IPC.** Less divergence (20.4 → 22.9
    active threads) packs the same thread work into ~11 % fewer warp instructions. Better issue
    (no-eligible 42 → 32 %) gives the rest.
  - **Occupancy rose only 5 points.** Block granularity is a smaller part of the gain than expected.
    It is not the main mechanism.
  - **L2 traffic went *up* 2.65×.** One 1,024-thread block launches 32 warps together that scan the
    sphere list in step, so one miss serves many warps. Small blocks start and finish at different
    times, so their warps are at different points in the scan and share less of L1.
    Faster anyway: L2 is still nearly idle (0.59 %), and the latency is hidden better.
  - *Stalls 32×32 → 8×8* (cycles per issued instruction; abs. = × instructions, −11 %):

    | Reason | 32×32 | 8×8 | abs. change |
    |---|---|---|---|
    | long_scoreboard | 2.67 | 2.41 | −19 % |
    | wait | 2.41 | 2.40 | −11 % (only fewer instructions) |
    | not_selected | 2.24 | 1.64 | −35 % |
    | **math_pipe_throttle** | 1.77 | **1.12** | **−43 %** |
    | no_instruction | 0.29 | 0.50 | +54 % (small) |
    | branch_resolving | 0.57 | 0.56 | −12 % |
    | total | 11.04 | 9.70 | −21 % |

    - **Lockstep vs staggered warps.** The 32 warps of one big block run in step: they all want
      the FMA pipe at the same time (math_pipe_throttle, not_selected) and all wait for memory at
      the same time (no eligible warp). Staggered small blocks mix phases, so pipe use is smoother:
      math throttle −43 %, not_selected −35 %. The cost: less L1 sharing (L2 bytes 2.65×). Here
      the smoother issue wins.
    - long_scoreboard −19 % despite 2.65× L2 traffic: the latency is hidden better.
    - **wait is unchanged per instruction:** the serial `oc → dot → disc` chain does not care about
      block shape. It is now the largest stall together with long_scoreboard: an ILP target.
    - no_instruction +54 %: more warps at different points in the code ⇒ more instruction-fetch
      misses. Minor (0.5 cycles).
  - FMA is now the top pipe at 64 %; ncu: "Compute is more heavily utilized than Memory".
    Remaining ncu estimates: ~20 % divergence, ~20 % occupancy (upper bounds).
  - nsys single run: 80.2 ms (vs 46.2 ms in `compare_blocks.sh`). Not used, as before.
- *Timing sweep split vs baseline* (32×32 blocks, `results/compare_sizes_split.txt`; image check
  IDENTICAL; `LDG.E.128` count: baseline 0, split 8):

  | N | baseline | split | speedup | base ms/sph | split ms/sph |
  |---|---|---|---|---|---|
  | 400 | 4.363 | 4.013 | 1.09× | – | – |
  | 1000 | 10.072 | 9.725 | 1.04× | 0.01009 | 0.00956 |
  | 1364 | 13.821 | 13.239 | 1.04× | 0.01079 | 0.00935 |
  | 1500 | 15.605 | 14.644 | 1.07× | 0.01382 | 0.00998 |
  | 1700 | 19.592 | 16.613 | 1.18× | 0.01993 | 0.00984 |
  | 1819 | 24.657 | 17.797 | 1.39× | 0.04256 | 0.00995 |
  | 2000 | 29.076 | 19.622 | 1.48× | 0.02519 | 0.01002 |
  | 2500 | 37.320 | 24.980 | 1.49× | 0.01649 | 0.01072 |
  | 3000 | 45.991 | 30.726 | **1.50×** | 0.01734 | 0.01149 |
  | 3300 | 51.298 | 34.464 | 1.49× | 0.01769 | 0.01246 |
  | 3600 | 57.085 | 38.768 | 1.47× | 0.01929 | 0.01435 |
  | 3800 | 61.219 | 42.230 | 1.45× | 0.02067 | 0.01731 |
  | 4000 | 64.538 | 49.558 | 1.30× | 0.01659 | **0.03664** |
  | 5000 | 84.007 | 66.347 | 1.27× | 0.01947 | 0.01679 |

  (Rows 600, 800, 1200, 1365, 1820 in the raw file.)
  - **While both versions fit in L1 (N ≤ 1,364) the split gains only 3–9 %.** Loads were not the
    bottleneck there.
  - **Split's L1 cliff: 3,800 → 4,000** (0.0366 ms/sph), exactly at 64 KB / 16 B = 4,096 spheres. It
    moved from ~1,820 by the predicted 2.25×.
  - Unlike the baseline, the split degrades *gradually* before the cliff: ms/sph creeps up from
    ~2,500 (40 KB, ~60 % of L1). The baseline was flat to ~1,364 and fell off from ~1,500.
  - **Biggest gain: 1.45–1.50× for N = 2,000–3,800**, where the split still fits L1 and the
    baseline doesn't. Past both cliffs (N=5000) both are L2-resident: 1.27× (ncu 1.19×).
  - Combined with 8×8 blocks, at N=5000: 84.0 / 46.2 ≈ **1.8× over the baseline** (different
    Colab sessions, so approximate).

*Next levers:* for N > ~4,000, split + shared memory / tiling (64 KB of shared memory holds
4,096 `float4` vs 1,820 `Sphere`); ILP (2 spheres per iteration; 20 spare registers before
occupancy drops); divergence (22.9 / 32 active threads at 8×8); FP32 instruction count (FMA now 64 %).

## 11. Optimization 3 — ILP, 2 spheres per loop iteration (`multi_reflections_2sphere.cu`)

*Change* (on top of the split): `closestHit` tests spheres `i` and `i+1` in one pass. Both
`hitSphere` calls come first, then the two `closest` updates (in order, so ties resolve as
before), then one shadow exit. A single test after the loop handles an odd `scene_size`.
Motivation (§10): wait (2.40) + long_scoreboard (2.41) ≈ half the stall cycles at 8×8. In-order
issue per warp means sphere `i+1` cannot start while sphere `i`'s chain stalls, unless both are in
the same loop body.

*Timing sweep* (`OPT=multi_reflections_2sphere.cu bash compare_sizes.sh`, 32×32 blocks,
`results/compare_sizes_2sphere.txt`), against the split run of §10 (different Colab session):

| N | split | 2sphere | 2sphere vs split | 2sphere vs baseline (same session) |
|---|---|---|---|---|
| 400 | 4.013 | 4.030 | 1.00× | 1.04× |
| 1000 | 9.725 | 9.840 | 0.99× | 1.02× |
| 2000 | 19.622 | 19.943 | 0.98× | 1.45× |
| 3000 | 30.726 | 31.010 | 0.99× | 1.47× |
| 3600 | 38.768 | 38.406 | 1.01× | 1.45× |
| 4000 | 49.558 | 48.938 | 1.01× | 1.30× |
| 5000 | 66.347 | 67.890 | 0.98× | 1.20× |

- Image check: IDENTICAL. Static `LDG.E.128` count: **2sphere 6 vs split 8** (fewer, not more).
  Hypothesis: the compiler had already unrolled the split's single-sphere loop ×4 at each of the 2
  inlined call sites (2 × 4 = 8), so loads for several spheres were already issued together. The manual
  pair loop + tail gives 2 × (2 + 1) = 6. If so, the ILP was already there.
  **SASS (`cuobjdump -sass | grep -n LDG.E.128`): unroll confirmed, overlap refuted.**
  - Split: `[R20]`, `[R20+0x10]`, `+0x20`, `+0x30` = spheres i..i+3 (×4 unroll), plus a ×3 group
    and a remainder load. But they are **~110 SASS lines apart and all load into R4**.
  - 2sphere: `[R8]` and `[R8+0x10]`, also ~108 lines apart, also into R4, then the tail `[R2]`.
  - So in *both* versions each load is followed by that sphere's whole test before the next load
    is issued (same destination registers ⇒ strictly one sphere in flight). The manual pairing did
    not change the schedule, which explains the null result.
  - Likely cause: `hitSphere` is not straight-line code. `if (disc < 0) return -1`, the `sqrt`, and
    the IEEE division `/(2*a)` (reciprocal + slow-path branch) put branches inside every sphere test,
    and the compiler does not move the next load above them.
  - Options: (a) load both `float4`s into locals at the top of the pass, before any `hitSphere`
    math; (b) make the test branch-free (select instead of early return, no division in the loop);
    (c) software prefetch: load sphere i+1 while testing sphere i (register double-buffering).
- **No measurable gain at 32×32**: within ±2 %, about the session-to-session noise (the baseline
  itself moved 84.0 → 81.5 ms at N=5000 between the two sessions).
- Cliff slightly earlier than the split (3,600 → 3,800 jumps to 0.0246 ms/sph), possibly noise.
- Not yet known *why*. To check: the image check / `LDG.E.128` lines; SASS of the loop (are the
  two `LDG.E.128` adjacent, or still separated?); `hitSphere`'s `if (disc < 0) return -1` is a
  branch at the end of each chain; per-warp stalls only cost time when no other warp is eligible.
  Next: 8×8 block sweep and an 8×8 profile vs `profiles_split_8x8`.

### 11.1 Software prefetch (`multi_reflections_prefetch.cu`)

*Change* (on top of the split, single-sphere loop): `curr` holds sphere `i`; at the top of each pass,
`next = active_scene[i+1]` (guarded) is issued *before* `hitSphere(curr)`; `curr = next` at the end.

*SASS:* the compiler still unrolls ×4, but the loads now go to **rotating registers** (R12, R16, R20,
R24, …) and are predicated (`@!P0`, the bounds guard). Each sphere's data is loaded one full test
(~110 SASS lines) before its first use. The overlap is now in the schedule.

*Timing* (32×32; `results/compare_sizes_prefetch.txt`; image IDENTICAL; 8 `LDG.E.128`).
Sessions differ (baseline at N=5000: 84.0 ms in the split session, 78.1 ms here), so compare
**speedup over the same-session baseline**:

| N | split vs base | prefetch vs base | prefetch ms/sph |
|---|---|---|---|
| 400 | 1.09× | 0.98× | – |
| 1000 | 1.04× | 0.97× | 0.01002 |
| 1819 | 1.39× | 1.30× | 0.01203 |
| 2000 | 1.48× | 1.40× | 0.01058 |
| 3000 | 1.50× | 1.40× | 0.01158 |
| 3600 | 1.47× | 1.38× | 0.01251 |
| 3800 | 1.45× | 1.37× | 0.01459 |
| 4000 | 1.30× | **1.35×** | **0.01660** (split: 0.03664) |
| 5000 | 1.27× | 1.28× | 0.01550 |

- **While the scene fits in L1, prefetch is ~6–9 % slower than the split.** L1 hits are short, so there
  is little latency to hide, and the extra predicated load, guard and register copies cost instructions.
- **The split's cliff at 3,800 → 4,000 is gone**: 0.0166 ms/sph vs 0.0366. Where misses start, the
  prefetch hides them. At N=4000, 1.35× vs 1.30×.
- At N=5000 (fully L2-resident): equal (1.28× vs 1.27×). Expected more. Needs a profile.

*Block sweep* (`results/compare_blocks_prefetch.txt`, all shapes IDENTICAL):

| N | 32×32 | 8×8 | 8×8 gain | split 8×8 gain (§10) |
|---|---|---|---|---|
| 400 | 4.231 | 3.227 | 1.31× | 1.30× |
| 3000 | 31.797 | 24.561 | 1.29× | 1.27× |
| 5000 | 60.426 | 49.607 | **1.22×** | **1.39×** |

- At N=5000, small blocks help the prefetch version much less (1.22× vs 1.39×), and prefetch 8×8
  (49.6 ms) is not faster than split 8×8 (46.2 ms, other session).
- **TLP and ILP hide the same latency.** With 8×8 blocks, other warps already cover most of the load
  wait. Once one mechanism has hidden it, the other adds little but keeps its instruction overhead.
*Same-session comparison, split vs prefetch* (`BASE=multi_reflections_split.cu
OPT=multi_reflections_prefetch.cu [BLOCK="8 8"] bash compare_sizes.sh`; images IDENTICAL;
`results/compare_sizes_prefetch.txt`, `results/compare_sizes_prefetch_8x8.txt`). Prefetch speedup over split:

| N | 32×32 | 8×8 |
|---|---|---|
| 400–3600 | 0.93–0.98× | 0.93–0.97× |
| 3800 | 0.96× | 0.93× |
| 4000 | 1.00× | 0.94× |
| 5000 | **1.03×** | 0.96× |

- **Prefetch is slower almost everywhere: ~5 % at 32×32, ~5 % at 8×8.** It only breaks even or wins
  slightly at 32×32 for N ≥ 4,000 (past the split's L1 cliff, few warps to hide misses).
- **Conclusion: drop it.** TLP (8×8 blocks) hides the same load latency without adding instructions.
  At 8×8 the kernel is closer to issue-bound (issue slots 62 %, FMA 64 %), so every extra
  instruction costs time directly. The cross-session estimates above (1.35× vs 1.30× at N=4000)
  overstated prefetch. Same-session runs are needed for differences of a few %.
- Side result: at 8×8 the split's own L1 cliff is mild (ms/sph 0.0075–0.0087 → 0.0106–0.0139 for
  N ≥ 3,800, vs a 0.0248 jump at 32×32). More warps also hide most of the cliff.
- **Current best: split + 8×8 blocks** (47.1 ms at N=5000 in this session).
- Both ILP attempts (§11 and §11.1) failed for the same reason: neither removed work, and the
  latency they target is already mostly hidden by other warps. Next levers should *remove
  instructions*: fewer FP32 ops per sphere test (FMA pipe), divergence. Or remove misses for
  N > 4,000 (split + shared / tiling).

## 12. Optimization 4 — fewer FP32 instructions per sphere test (`multi_reflections_fma.cu`)

*Change* (on top of the split, `hitSphere` only):
- `.w` stores r² (computed on the host) instead of r.
- Half-b form with `a = dot(dir, dir) = 1`: every direction passed to `closestHit` is normalized
  (primary, `toLight`, reflected). `b' = dot(oc, dir)`, `disc' = b'² − c` (one FFMA),
  `t = −b' − √disc'` (no division, so no IEEE slow-path branch).
- `c = fmaf(oc.z, oc.z, fmaf(oc.y, oc.y, fmaf(oc.x, oc.x, −r²)))`: starting the sum from −r² folds
  the subtraction into the first FFMA. The compiler cannot reorder it itself (float addition is not
  associative, and the code is built without fast-math).

*FP32 instructions per sphere test* (ncu counters ÷ 5.74 G tests, N=5000, 8×8):

| | split | fma |
|---|---|---|
| FADD | 4 | 3 |
| FMUL | 3 | 1 |
| FFMA | 6 | 6 |
| **total** | **13** | **10** (−23 %) |

*Profile 8×8, N=5000* (`profiles_fma_8x8` vs `profiles_split_8x8`): duration 112.6 → 104.6 ms
(1.08×); instructions −10 %; FMA pipe 64 → 56 %; IPC 2.66 → 2.57 (the removed instructions were
partly ones that issued in otherwise idle slots); registers 44 → 43; occupancy 80.7 → 81.4 %.

*Timing* (`BASE=multi_reflections_split.cu OPT=multi_reflections_fma.cu BLOCK="8 8"`, same session,
`results/compare_sizes_fma_8x8.txt`): **1.06–1.13× over the split at every N** (N=5000: 45.8 →
41.9 ms).

*Image: not bit-identical.* 1,722 / 480,000 pixels (0.36 %) differ, max 212/255. Diff image
(`compare_images.py split.ppm fma.ppm diff.ppm`): the differing pixels are isolated single pixels
on the **silhouettes of small / distant spheres** and inside mirror reflections. They are not in blobs,
and not in shadows or on the near floor.
- Cause: `a = 1` holds only to ~1e-7. For a small far sphere, |oc|² / r² ≈ 100 / 0.0016 ≈ 6·10⁴,
  so that 1e-7 shifts the hit/miss boundary by ~0.3 % of the radius. A pixel whose ray grazes the
  rim can flip between sphere and background, a large colour jump in one pixel. The reordered sum
  for `c` adds rounding differences of the same kind.
- Neither image is "the correct one": both are within float precision of the true geometry. The
  difference is a sub-pixel move of silhouettes, the scale anti-aliasing works at.
- Optional attribution test: keep `a = dot(dir, dir)` (disc = b'² − a·c, divide by a) and see
  whether the diff mostly disappears. That costs one FMUL per test, plus a division on hits only.

## 13. Open items

- [x] ncu **Source** page: stalls are ~98 % in the sphere loop; 4 scalar `LDG` per sphere (§3.6).
- [x] Warning triangles and line-50 stall breakdown (§3.6).
- [x] Hierarchical roofline dot colours: L1 purple, L2 orange, DRAM teal (§10).
- [ ] L1 byte count the roofline uses: 4.0× `l1tex__t_bytes` for the baseline, 12.4× for the split.
      Check the thread-bytes hypothesis on the Raw page (§10).
- [x] Profile baseline at N=5000 (§9).
- [x] Re-run nsys with a warm-up: added to `profile.sh`; still ~1.28× `compare_sizes.sh` at N=5000 (§9).
- [ ] Optional: profile baseline N=1819 (the thrashing peak) and shared N=1819 (see §8).
- [x] Profile split at N=5000 (§10).
- [x] Timing sweep split vs baseline (§10).
- [x] Block-shape sweep and 8×8 profile (§10).
- [x] Stall breakdown for the 8×8 profile (§10).
- [ ] Re-run `compare_sizes.sh` with the best block shape (needs a BX/BY pass-through or 8×8 as default).
