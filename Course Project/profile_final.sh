#!/bin/bash
set -e

# Usage: bash profile_final.sh
#   Profiles every flat version once (nsys + ncu full set + roofline counters, via profile.sh)
#   with the final measurement method (ncu profiles only the timed launch, after the
#   in-process warm-up). Depth 10, threshold 0.01. Reports go to profiles_final/<scene>/<version>/.
#   SCENES: scenes to profile (default: all three). On the main scene every version is
#   profiled; on the other scenes only the main steps (OTHER_VERSIONS).
#   Each profile takes ~1-4 min (ncu replays the kernel ~40 times at a locked 585 MHz):
#   ~30 min for the main scene, ~10 min for each other scene.
#   Download with:  zip -qr profiles_final.zip profiles_final
SCENES=${SCENES:-"scene.h scene_adversarial.h kaushik_scene.h"}
# version:block:N   (shared only fits 1,819 spheres, so it is profiled at N=400 with the baseline)
MAIN_VERSIONS=${MAIN_VERSIONS:-"timed:32x32:400 shared:32x32:400 timed:32x32:5000 split:32x32:5000 split:8x8:5000 2sphere:8x8:5000 prefetch:8x8:5000 fma:8x8:5000"}   # fp16:8x8:5000 later
OTHER_VERSIONS=${OTHER_VERSIONS:-"timed:32x32:5000 split:32x32:5000 split:8x8:5000 fma:8x8:5000"}   # fp16:8x8:5000 later

for S in $SCENES; do
    NAME=$(basename "$S" .h)
    if [ "$S" = scene.h ]; then VERSIONS=$MAIN_VERSIONS; else VERSIONS=$OTHER_VERSIONS; fi
    for V in $VERSIONS; do
        IFS=: read -r VER BLOCK N <<< "$V"
        OUT=profiles_final/$NAME/${VER}_${BLOCK}_N$N
        echo; echo "===== $(date +%H:%M:%S) $S: $VER $BLOCK N=$N -> $OUT ====="
        SCENE=$S SRC=multi_reflections_$VER.cu OUT=$OUT bash profile.sh "$N" 10 0.01 ${BLOCK/x/ }
    done
done
echo "Profiles in profiles_final/. Download with:  zip -qr profiles_final.zip profiles_final"
