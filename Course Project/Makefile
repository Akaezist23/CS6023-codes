# CUDA ray tracer (CS6023 course project): build every version used in the report.
#
#   make              build the CPU starter and all CUDA versions into bin/
#   make run          render the default scene with the best flat version and the BVH
#   make images       render the images of the report into images/
#   make clean        remove bin/, images/ and image.ppm
#
# Options (on the command line, e.g. "make ARCH=-arch=sm_75 SCENE=scene.h"):
#   ARCH    GPU architecture flag; empty = nvcc's default (all measurements: Tesla T4, sm_75)
#   SCENE   scene header in src/ (scene.h, kaushik_scene.h, scene_adversarial.h); empty = scene.h
#
# Every CUDA program writes image.ppm and prints its kernel time (CUDA events, after an untimed
# warm-up launch). Arguments:
#   bin/baseline [W H]                                   3 spheres, no reflections (the port)
#   bin/<version> [N depth threshold BX BY W H]          all multi-bounce versions
#   bin/cpu_raytracer                                    the given CPU starter (3 spheres)

NVCC    ?= nvcc
CXX     ?= g++
ARCH    ?=
SCENE   ?=
NVFLAGS  = -O2 -std=c++11 $(ARCH) $(if $(SCENE),-DSCENE_FILE='"$(SCENE)"')

# Multi-bounce versions, in the order of the report (src/multi_reflections_<name>.cu)
VERSIONS = timed shared split 2sphere prefetch fma fp16 bvh
# Intermediate versions not discussed in the report (built with "make extras")
EXTRAS   = naive threshold bvh_bare bvh_reps

BINS = bin/cpu_raytracer bin/baseline $(addprefix bin/,$(VERSIONS))

.PHONY: all extras run images clean
all: $(BINS)
extras: $(addprefix bin/,$(EXTRAS))

bin:
	mkdir -p bin

bin/cpu_raytracer: starter_raytracer_cpu.cpp | bin
	$(CXX) -O2 -o $@ $<

bin/baseline: src/baseline.cu | bin
	$(NVCC) $(NVFLAGS) -o $@ $<

bin/%: src/multi_reflections_%.cu src/scene.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $<

# Best flat version (8x8 blocks) and the BVH on the full 5,000-sphere scene
run: bin/fma bin/bvh
	./bin/fma 5000 10 0.01 8 8
	./bin/bvh 5000 10 0.01 8 8

# Images of the report: the port (3 spheres), reflection depths 0/1/3/10 at 400 spheres,
# and the 5,000-sphere scene (other scenes: make images SCENE=... after make clean)
images: bin/baseline bin/fma
	mkdir -p images
	./bin/baseline && mv image.ppm images/port_3_spheres.ppm
	for d in 0 1 3 10; do ./bin/fma 400 $$d 0.01 8 8 && mv image.ppm images/depth_$$d.ppm; done
	./bin/fma 5000 10 0.01 8 8 && mv image.ppm images/scene_5000.ppm
	@echo "Images in images/ (PPM; convert with e.g. python3 -c \"from PIL import Image; ...\")"

clean:
	rm -rf bin images image.ppm
