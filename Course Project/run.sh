#!/bin/bash
set -e

# Usage: bash run.sh [depth] [threshold]   (reflection depth, default 0; energy threshold, default 0.01)
DEPTH=${1:-0}
THRESHOLD=${2:-0.01}

# Build
# g++ -O2 -o raytracer_cpu starter_raytracer_cpu.cpp
# nvcc baseline.cu -o baseline -O2 -std=c++11
nvcc multi_reflections_2.cu -o multi_reflections -O2 -std=c++11

# Run (writes image.ppm)
# ./raytracer_cpu
# ./baseline
./multi_reflections "$DEPTH" "$THRESHOLD"

# rm raytracer_cpu
# rm baseline
rm multi_reflections

# Convert image.ppm -> image.png (PIL is preinstalled on Colab)
python3 -c "from PIL import Image; Image.open('image.ppm').save('image.png')"
echo "Wrote image.png"

# A shell started with `!bash run.sh` can't draw into the Colab output cell,
# so display image.png from Python in the same cell:
#   !bash run.sh 3
#   from IPython.display import Image, display; display(Image('image.png'))


