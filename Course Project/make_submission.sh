#!/bin/bash
set -e

# Usage: bash make_submission.sh
#   Builds the submission zip (CS6023_project_cs23b078.zip) from a clean copy of the files the
#   submission needs: Makefile, README, report, sources, scripts and the results behind the report.
#   Put the compiled report.pdf (from Overleaf) in this folder first.
NAME=CS6023_project_cs23b078
STAGE=$(mktemp -d)
DEST=$STAGE/$NAME
mkdir -p "$DEST/results"

[ -f report.pdf ] || echo "WARNING: report.pdf not found; the zip will only contain report.tex"

cp Makefile README.md report.tex starter_raytracer_cpu.cpp "$DEST/"
[ -f report.pdf ] && cp report.pdf "$DEST/"
cp -r src "$DEST/"
cp benchmark_final.sh compare_all.sh compare_blocks.sh compare_cpu.sh compare_depth.sh \
   compare_resolution.sh compare_sizes.sh sweep_all.sh run.sh profile.sh profile_final.sh \
   nsys_profile.sh "$DEST/"
cp compare_images.py table_to_csv.py merge_sweep.py ncu_to_csv.py make_adversarial_scene.py "$DEST/"
cp -r results/final results/depth_images "$DEST/results/"
rm -rf "$DEST/results/final/scene/sweep_all_10pts" "$DEST/results/final/scene/sweep_all_10pts.csv"

rm -f "$NAME.zip"
(cd "$STAGE" && zip -qr "$OLDPWD/$NAME.zip" "$NAME")
rm -rf "$STAGE"
echo "Wrote $NAME.zip ($(du -h "$NAME.zip" | cut -f1))"
