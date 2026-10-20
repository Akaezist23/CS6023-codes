# CLAUDE.md

## Rules
- **Do NOT edit any `.cpp` files** (e.g. `starter_raytracer_cpu.cpp`). They are read-only reference code.
  Only read and explain them.

## Context
- Course project for CS6023 (GPU Programming): CUDA Ray Tracing (spec in `kla2-rt.pdf`).
- `starter_raytracer_cpu.cpp` is the given CPU ray tracer (3 spheres + checkerboard floor, diffuse shading, hard shadows).
  Build/run via `run.sh` → writes `image.ppm`.
- Current mode: walking through the starter code with the user, **one code chunk per exchange**.
  Keep explanations short and focused; don't dump monolithic responses.
- The spec asks that LLM use be limited to explaining snippets, clearing doubts, and discussing ideas,
  so explain and discuss rather than writing the CUDA port for the user.
