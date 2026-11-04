// ============================================================================
//  Basic CPU ray tracer  --  starter code
// ============================================================================
//  Renders a few spheres on a checkerboard floor, with simple diffuse shading
//  and hard shadows. Single-threaded, plain C++. No external libraries.
//
//  This is your STARTING POINT -- it is short on purpose.
//  Later you will (a) add reflections, (b) port it to CUDA, and (c) make it fast.
//
//  Build:  g++ -O2 -o raytracer_cpu starter_raytracer_cpu.cpp
//  Run:    ./raytracer_cpu                 (writes image.ppm)
//  View:   most image viewers open .ppm; or convert with:  convert image.ppm image.png
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// ---- A tiny 3D vector, used for points, directions, and colors (r,g,b) -----
struct Vec3 {
    float x, y, z;
    __device__ __host__ Vec3(float a = 0, float b = 0, float c = 0) : x(a), y(b), z(c) {}
};
__device__ __host__ Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ __host__ Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ __host__ Vec3 operator*(Vec3 a, float s) { return Vec3(a.x * s, a.y * s, a.z * s); }
__device__ __host__ float dot(Vec3 a, Vec3 b)  { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ __host__ Vec3  normalize(Vec3 a)    { float len = std::sqrt(dot(a, a)); return a * (1.0f / len); }

// ---- A sphere: a center, a radius, and a color ----------------------------
struct Sphere {
    Vec3  center;
    float radius;
    Vec3  color;
    float reflectivity; // 0 = diffuse, 1 = perfect mirror
    bool  isFloor;   // if true, we draw a checkerboard instead of a solid color
};

struct __align__(16) bvhNode {
    float minx, miny, minz;
    int leftOrFirst; // gives you indices of children (if inner node) or first sphere (if leaf node)
    float maxx, maxy, maxz;
    int count; // 0 if inner node, > 0 to indicate #spheres in leaf node
};

#include "scene.h"

__device__ float hitSphere(const float4& s, Vec3 origin, Vec3 dir) {
    Vec3  center(s.x, s.y, s.z);
    float radius2 = s.w;
    Vec3  oc = origin - center;
    float a = dot(dir, dir);
    float b = dot(oc, dir);
    float c = fmaf(oc.z, oc.z, fmaf(oc.y, oc.y, fmaf(oc.x, oc.x, -radius2)));
    float disc = b * b - a * c; 
    if (disc < 0) return -1.0f;                 // no real solution -> miss
    float t = (-b - std::sqrt(disc))/a; // nearest intersection
    return (t > 0.001f) ? t : -1.0f;
}

// Find the closest sphere the ray hits. Returns its index (or -1) and the distance.
__device__ int closestHit(Vec3 origin, Vec3 dir, float4* active_scene, int scene_size , float& tHit, bool shadow = false) {
    int   best = -1;
    float closest = 1e30f;
    for (int i = 0; i < scene_size; i++) {
        float t = hitSphere(active_scene[i], origin, dir);
        if (shadow && t > 0) return i; // we don't need the distance for shadows
        if (t > 0 && t < closest) { closest = t; best = i; }
    }
    tHit = closest;
    return best;
}

// Compute the color seen along one ray.
__device__ Vec3 rayColor(Vec3 origin, Vec3 dir, Sphere* scene, float4* active_scene, int scene_size, int depth = 0, float threshold = 0.01f) {
    // TRY IT: move the light around and see how shading/shadows change.
    // x = right, y = up, z = toward the camera.
    Vec3  lightPos(4, 6, 2);
    float tHit;
    Vec3 base(0.2f, 0.2f, 0.2f);
    int hit = -1;
    Vec3 p, n;
    Vec3 color(0.f, 0.f, 0.f);
    float curr = 1;
    do {
        depth--;
        hit = closestHit(origin, dir, active_scene, scene_size, tHit);
    
        if (hit < 0) {
            // Miss: draw a simple sky gradient (white near horizon -> blue up top).
            float t = 0.5f * (dir.y + 1.0f);
            color = color + (Vec3(1, 1, 1) * (1 - t) + Vec3(0.4f, 0.6f, 1.0f) * t) * curr;
            break;
        }
        // The point we hit, and the surface normal there.
        p = origin + dir * tHit;
        n = normalize(p - scene[hit].center);
        // Base color: checkerboard for the floor, solid color otherwise.
        base = scene[hit].color;
        if (scene[hit].isFloor) {
            int cx = (int)std::floor(p.x), cz = (int)std::floor(p.z);
            bool white = ((cx + cz) & 1) == 0;
            base = white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f);
        }
        float k = scene[hit].reflectivity;
        
        Vec3  toLight = normalize(lightPos - p);
        float diffuse = std::fmax(0.0f, dot(n, toLight));
        
        // Hard shadow: shoot a ray toward the light; if it hits something, we're shadowed.
        float shadowT;
        int   blocked = closestHit(p + n * 0.001f, toLight, active_scene, scene_size, shadowT, true);
        float shade = (blocked >= 0) ? 0.3f : 1.0f;
        
        float brightness = 0.15f + diffuse * shade;   // 0.15 = a little ambient light
        Vec3 local = base * brightness;
        color = color + local * curr * (1.0f - k);
        curr *= k;
        origin = p + n * 0.001f; // move origin slightly above the surface to avoid self-intersection
        dir = normalize(dir - n * 2 * dot(dir, n)); // direction of the reflected ray
        
    } while (depth >= 0 && curr > threshold); 
       
    return color;
}

__global__ void renderKernel(unsigned char* image, Sphere* scene, float4* active_scene, int scene_size , int W, int H, Vec3 cameraPos, float aspect, int depth, float threshold) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= W || py >= H) return;

    // Map pixel (px,py) to a direction through the image plane.
    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3 dir = normalize(Vec3(u, v, -1.5f));

    Vec3 color = rayColor(cameraPos, dir, scene, active_scene, scene_size, depth, threshold);

    // Store the color as 3 bytes (clamp to [0,1] first).
    int idx = (py * W + px) * 3;
    image[idx + 0] = (unsigned char)(fmin(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fmin(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fmin(color.z, 1.0f) * 255);
}

bvhNode findBox(int first, int cnt, std::vector<int>& idx) {
    bvhNode box;
    box.minx = box.miny = box.minz = 1e30f;
    box.maxx = box.maxy = box.maxz = -1e30f;
    for (int i = first; i < first + cnt; ++i) {
        int sphereIdx = idx[i];
        const Sphere& s = scene[sphereIdx];
        box.minx = std::fmin(box.minx, s.center.x - s.radius);
        box.miny = std::fmin(box.miny, s.center.y - s.radius);
        box.minz = std::fmin(box.minz, s.center.z - s.radius);
        box.maxx = std::fmax(box.maxx, s.center.x + s.radius);
        box.maxy = std::fmax(box.maxy, s.center.y + s.radius);
        box.maxz = std::fmax(box.maxz, s.center.z + s.radius);
    }
    return box;
}

void build_bvh(int i, int first, int cnt, int& usedNodes, std::vector<int>& idx, std::vector<bvhNode>& nodes) {
    bvhNode box = findBox(first, cnt, idx);
    nodes[i] = box;
    if (cnt <= 4) {
        nodes[i].leftOrFirst = first;
        nodes[i].count = cnt;
        return;
    }
    float ex = box.maxx - box.minx, ey = box.maxy - box.miny, ez = box.maxz - box.minz;
    int axis = 0;
    if (ey > ex) axis = 1;
    if (ez > ex && ez > ey) axis = 2;

    auto cmp = [axis](int a, int b) {
        Vec3& ca = scene[a].center;
        Vec3& cb = scene[b].center;
        float va = (axis == 0) ? ca.x : (axis == 1) ? ca.y : ca.z;
        float vb = (axis == 0) ? cb.x : (axis == 1) ? cb.y : cb.z;
        return va < vb;
    };
    int half = cnt / 2;
    std::nth_element(idx.begin() + first, idx.begin() + first + half, idx.begin() + first + cnt, cmp);
    nodes[i].count = 0; 
    int left = usedNodes;
    usedNodes += 2;
    nodes[i].leftOrFirst = left;
    build_bvh(left, first, half, usedNodes, idx, nodes);
    build_bvh(left + 1, first + half, cnt - half, usedNodes, idx, nodes);
}

// Command-line settings: ./multi_reflections <N> <depth> <threshold> <BX> <BY> <W> <H>
struct Args {
    int   N;          // render the floor + the first N spheres
    int   depth;      // max reflection depth
    float threshold;  // stop once ray energy drops below this
    int   bx, by;     // block size (threads) in x and y
    int   W, H;       // image size in pixels
};

// Fills `a` from argv (missing args get defaults). Returns false on invalid input.
bool parseArgs(int argc, char** argv, Args& a) {
    int maxN = (int)scene.size() - 1;                            // scene[0] is the floor
    a.N = (argc > 1) ? std::atoi(argv[1]) : maxN;
    if (a.N < 1) { fprintf(stderr, "N must be >= 1\n"); return false; }
    if (a.N > maxN) a.N = maxN;
    a.depth = (argc > 2) ? std::atoi(argv[2]) : 10;
    if (a.depth < 0) { fprintf(stderr, "depth must be >= 0\n"); return false; }
    a.threshold = (argc > 3) ? (float)std::atof(argv[3]) : 0.01f;
    if (a.threshold < 0.0f || a.threshold > 1.0f) { fprintf(stderr, "threshold must be in [0, 1]\n"); return false; }
    a.bx = (argc > 4) ? std::atoi(argv[4]) : 32;
    a.by = (argc > 5) ? std::atoi(argv[5]) : 32;
    if (a.bx < 1 || a.by < 1 || a.bx * a.by > 1024) { fprintf(stderr, "need BX, BY >= 1 and BX * BY <= 1024\n"); return false; }
    a.W = (argc > 6) ? std::atoi(argv[6]) : 800;
    a.H = (argc > 7) ? std::atoi(argv[7]) : 600;
    if (a.W < 1 || a.H < 1) { fprintf(stderr, "need W, H >= 1\n"); return false; }
    return true;
}

int main(int argc, char** argv) {
    static_assert(sizeof(bvhNode) == 32, "Node must be two 16-byte halves"); //we'll remove this at the end

    Args args;
    if (!parseArgs(argc, argv, args)) return 1;
    int   N = args.N, depth = args.depth, bx = args.bx, by = args.by, W = args.W, H = args.H;
    float threshold = args.threshold;

    // x = right, y = up, z = toward the viewer (spheres sit near z = -1).
    Vec3  cameraPos(0.0f, 1.0f, 3.0f);
    int   sceneSize = N + 1;

    std::vector<unsigned char> image(W * H * 3);
    std::vector<int> idx(sceneSize);
    std::vector<Sphere> bvh_scene(sceneSize);
    std::vector<bvhNode> nodes(sceneSize * 2); // worst case: every sphere is a leaf node, so we need 2*sceneSize nodes
    int usedNodes = 1;
    for (int i = 0; i < sceneSize; ++i) idx[i] = i;
    build_bvh(0, 1, sceneSize - 1, usedNodes, idx, nodes);
    for (int i = 0; i < sceneSize; ++i) bvh_scene[i] = scene[idx[i]];

    Sphere* d_scene;
    float4* d_active_scene;
    unsigned char* d_image;
    cudaMalloc(&d_image, image.size());
    cudaMalloc(&d_scene, sceneSize * sizeof(Sphere));
    cudaMemcpy(d_scene, bvh_scene.data(), sceneSize * sizeof(Sphere), cudaMemcpyHostToDevice);
    float aspect = (float)W / H;

    std::vector<float4> active_scene(sceneSize);
    for (int i = 0; i < sceneSize; ++i) {
        const Sphere& s = bvh_scene[i];
        active_scene[i] = make_float4(s.center.x, s.center.y, s.center.z, s.radius * s.radius);
    }
    cudaMalloc(&d_active_scene, sceneSize * sizeof(float4));
    cudaMemcpy(d_active_scene, active_scene.data(), sceneSize * sizeof(float4), cudaMemcpyHostToDevice);

    // Time just the kernel with CUDA events.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    renderKernel<<<dim3((W + bx - 1) / bx, (H + by - 1) / by), dim3(bx, by)>>>(d_image, d_scene, d_active_scene, sceneSize, W, H, cameraPos, aspect, depth, threshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernelMs = 0.0f;
    cudaEventElapsedTime(&kernelMs, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(image.data(), d_image, image.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_scene);
    cudaFree(d_active_scene);


    // Write a PPM image file (a very simple image format).
    FILE* f = fopen("image.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image.data(), 1, image.size(), f);
    fclose(f);
    printf("Wrote image.ppm (%dx%d, N %d, depth %d, threshold %g, block %dx%d)\n", W, H, N, depth, threshold, bx, by);
    printf("Kernel time: %.3f ms\n", kernelMs);
    return 0;
}

