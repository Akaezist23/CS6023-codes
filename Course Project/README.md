# CUDA Ray Tracer (CS6023 course project)

The report is `report.pdf` (source: `report.tex`).

## Build and run

Needs the CUDA toolkit (`nvcc`) and `g++`. All measurements were made on a Tesla T4 (sm_75).

```bash
make                 # builds the CPU starter and every CUDA version into bin/
make run             # best flat version and BVH on the 5,000-sphere scene: prints both kernel times
make images          # renders the report's images into images/ (PPM)
make ARCH=-arch=sm_75 SCENE=kaushik_scene.h   # options: target GPU, scene (after "make clean")
```

Every program writes `image.ppm` and prints its kernel time. For example
`./bin/fma 5000 10 0.01 8 8` renders 5,000 spheres, depth 10, threshold 0.01, with 8x8 blocks;
`./bin/baseline` is the direct port of the CPU starter (`./bin/cpu_raytracer`), which renders the same
3-sphere image.

| Report section | Source | Data / script |
|---|---|---|
| 3 Port, CPU vs GPU | `src/baseline.cu`, `starter_raytracer_cpu.cpp` | `compare_cpu.sh`, `results/final/scene/compare_cpu.txt` |
| 4 Multiple bounces, depth, resolution | `src/multi_reflections_timed.cu`, `_fma.cu` | `compare_depth.sh`, `compare_resolution.sh`, `results/depth_images/` |
| 5 Scaling the scene | `src/scene.h` | `compare_sizes.sh`, `sweep_all.sh` |
| 6 Optimizations | `_shared`, `_split`, `_fma` (8x8 blocks) | `compare_all.sh`, `compare_blocks.sh`, `sweep_all.sh` |
| 7 Attempts that did not help | `_2sphere`, `_prefetch`, `_fp16` | `compare_all.sh`, `results/final/scene/fp16/` |
| 8 Profiles (stalls, roofline) | all of the above | `profile.sh`, `ncu_to_csv.py` |
| 9 BVH extension | `src/multi_reflections_bvh.cu` | `compare_all.sh`, `results/final/*/bvh/` |

All results behind the report's tables and plots are in `results/final/<scene>/` (text tables + CSV).

## Running the benchmarks

All measurements in `report.tex` were made on a Tesla T4 (Google Colab), with CUDA and `nvcc` on the path.
The scripts also need `python3` with Pillow (for the image checks) and, for profiling, Nsight Systems (`nsys`)
and Nsight Compute (`ncu`). Run every script from this folder; the sources are in `src/`.

## Programs

Each version in `src/` is a standalone program that renders `image.ppm` and prints its kernel time
(CUDA events, after an untimed warm-up launch in the same process):

```bash
nvcc src/multi_reflections_fma.cu -o rt -O2 -std=c++11
./rt N depth threshold BX BY W H      # e.g. ./rt 5000 10 0.01 8 8 800 600
```

- `N`: number of spheres (besides the floor) from the scene file
- `depth`: maximum reflection bounces
- `threshold`: a ray stops once its energy is below this value
- `BX BY`: block size
- `W H`: image size

All arguments are optional. To use another scene, pass `-DSCENE_FILE='"kaushik_scene.h"'` to `nvcc`;
the default scene is `scene.h`.

| Source | Version in the report |
|---|---|
| `baseline.cu` | direct CUDA port of `starter_raytracer_cpu.cpp` |
| `multi_reflections_timed.cu` | multi-bounce baseline |
| `multi_reflections_shared.cu` | scene in shared memory (at most 1,819 spheres) |
| `multi_reflections_split.cu` | sphere centre + radius in a separate `float4` array |
| `multi_reflections_2sphere.cu`, `_prefetch.cu` | the two attempts that did not help |
| `multi_reflections_fma.cu` | split + fewer FP instructions (best version, run with 8×8 blocks) |
| `multi_reflections_fp16.cu` | half-precision sphere storage (did not help) |
| `multi_reflections_bvh.cu` | BVH extension (`_bvh_bare.cu`: without nearer-first order and pruning) |
| `multi_reflections_naive.cu`, `_threshold.cu`, `_bvh_reps.cu` | intermediate versions, not in the report (`make extras`) |

## Benchmarks

Every script compiles what it needs, first checks that the versions render the same image, and then
times `RUNS` runs per point (default 10 in the final runs). Results go to `results/` as a text table
plus a CSV for plotting. Common options are set as environment variables: `SCENE`, `N`, `RUNS`, `OUT`.

| Script | Measures |
|---|---|
| `benchmark_final.sh` | everything in the report, in stages (`STAGES="cpu depth resolution all sizes blocks"`) |
| `compare_cpu.sh` | CPU starter vs its CUDA port |
| `compare_depth.sh [threshold]` | time vs reflection depth |
| `compare_resolution.sh` | time vs image size |
| `compare_sizes.sh` | time vs number of spheres, `BASE` vs `OPT` |
| `compare_blocks.sh` | block shapes for one version (`SRC`) |
| `compare_all.sh` | all versions at one `N`, runs interleaved |
| `sweep_all.sh` | `compare_all.sh` for several `N`, both scenes, merged into one CSV |
| `profile_final.sh`, `profile.sh` | Nsight Systems + Nsight Compute profiles |

To reproduce the report's measurements:

```bash
bash benchmark_final.sh      # ~1.5-2 h on a T4; results in results/final/<scene>/
bash sweep_all.sh 10 0.01    # all versions vs N, ~20-30 min
bash profile_final.sh        # profiles in profiles_final/<scene>/<version>/
```

Helpers:
- `compare_images.py a.ppm b.ppm [diff.ppm]` compares two images pixel by pixel.
- `table_to_csv.py` converts the text tables to CSV.
- `merge_sweep.py` merges the per-N tables of `sweep_all.sh`.
- `ncu_to_csv.py` extracts the stall breakdown and roofline numbers from Nsight Compute reports.

## LLM acknowledgement

The `Makefile`, the benchmark and profiling scripts (`*.sh`) and the Python helpers in this folder were written by
Claude (Anthropic's AI assistant). I decided what to measure and how, and Claude wrote the scripts.
`src/multi_reflections_fp16.cu`, a copy of the fp16 version with the types changed, was also written by Claude.
Claude was also used to explain code snippets and profiler output and to discuss optimization
ideas. The ray-tracer kernels and the optimizations themselves are my own work.
