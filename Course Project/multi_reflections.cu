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

// The whole scene is just a list of spheres. (The floor is a giant sphere.)
std::vector<Sphere> scene = {
    { Vec3(0, -1000.5f, -1), 1000.0f, Vec3(0.5f, 0.5f, 0.5f), 0.f, true  }, // floor
    { Vec3(0,     0.0f, -1),    0.5f, Vec3(0.9f, 0.2f, 0.2f), 0.5f, false }, // red
    { Vec3(-1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.7f, 0.3f), 0.5f, false }, // green
    { Vec3( 1.1f, 0.0f, -1),    0.5f, Vec3(0.2f, 0.4f, 0.9f), 0.5f, false }, // blue

};

// A ray is a point p(t) = origin + t * dir. This finds where (if anywhere) the
// ray hits the sphere, returning the nearest distance t in front of the camera,
// or -1 if it misses. (Solves a quadratic -- see any ray-sphere reference.)
__device__ float hitSphere(const Sphere& s, Vec3 origin, Vec3 dir) {
    Vec3  oc = origin - s.center;
    float a = dot(dir, dir);
    float b = 2.0f * dot(oc, dir);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0) return -1.0f;                 // no real solution -> miss
    float t = (-b - std::sqrt(disc)) / (2 * a); // nearest intersection
    return (t > 0.001f) ? t : -1.0f;
}

// Find the closest sphere the ray hits. Returns its index (or -1) and the distance.
__device__ int closestHit(Vec3 origin, Vec3 dir, Sphere* scene, int scene_size , float& tHit) {
    int   best = -1;
    float closest = 1e30f;
    for (int i = 0; i < scene_size; i++) {
        float t = hitSphere(scene[i], origin, dir);
        if (t > 0 && t < closest) { closest = t; best = i; }
    }
    tHit = closest;
    return best;
}

// Compute the color seen along one ray.
__device__ Vec3 rayColor(Vec3 origin, Vec3 dir, Sphere* scene, int scene_size, int depth = 0) {
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
        hit = closestHit(origin, dir, scene, scene_size, tHit);
    
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
        int k = scene[hit].reflectivity;
        
        Vec3  toLight = normalize(lightPos - p);
        float diffuse = std::fmax(0.0f, dot(n, toLight));
        
        // Hard shadow: shoot a ray toward the light; if it hits something, we're shadowed.
        float shadowT;
        int   blocked = closestHit(p + n * 0.001f, toLight, scene, scene_size, shadowT);
        float shade = (blocked >= 0) ? 0.3f : 1.0f;
        
        float brightness = 0.15f + diffuse * shade;   // 0.15 = a little ambient light
        Vec3 local = base * brightness;
        color = color + local * curr * (1.0f - k);
        curr *= k;
        origin = p + n * 0.001f; // move origin slightly above the surface to avoid self-intersection
        dir = normalize(dir - n * 2 * dot(dir, n)); // direction of the reflected ray
        
    } while (depth >= 0);
       
    return color;
}

__global__ void renderKernel(unsigned char* image, Sphere* scene, int scene_size , int W, int H, Vec3 cameraPos, float aspect, int depth) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= W || py >= H) return;

    // Map pixel (px,py) to a direction through the image plane.
    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3 dir = normalize(Vec3(u, v, -1.5f));

    Vec3 color = rayColor(cameraPos, dir, scene, scene_size, depth);

    // Store the color as 3 bytes (clamp to [0,1] first).
    int idx = (py * W + px) * 3;
    image[idx + 0] = (unsigned char)(fmin(color.x, 1.0f) * 255);
    image[idx + 1] = (unsigned char)(fmin(color.y, 1.0f) * 255);
    image[idx + 2] = (unsigned char)(fmin(color.z, 1.0f) * 255);
}

int main(int argc, char** argv) {
    // Reflection depth (number of bounces), from the command line: ./multi_reflections <depth>
    int depth = (argc > 1) ? std::atoi(argv[1]) : 0;
    if (depth < 0) { fprintf(stderr, "depth must be >= 0\n"); return 1; }

    const int W = 800, H = 600;
    std::vector<unsigned char> image(W * H * 3);
    Sphere* d_scene;
    unsigned char* d_image;
    cudaMalloc(&d_image, image.size());
    cudaMalloc(&d_scene, scene.size() * sizeof(Sphere));
    cudaMemcpy(d_scene, scene.data(), scene.size() * sizeof(Sphere), cudaMemcpyHostToDevice);
    // x = right, y = up, z = toward the viewer (spheres sit near z = -1).
    Vec3  cameraPos(0, 1, 3);
    float aspect = (float)W / H;

    renderKernel<<<dim3((W + 31) / 32, (H + 31) / 32), dim3(32, 32)>>>(d_image, d_scene, scene.size(), W, H, cameraPos, aspect, depth);
    // cudaDeviceSynchronize(); 
    cudaMemcpy(image.data(), d_image, image.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_scene);

    // Write a PPM image file (a very simple image format).
    FILE* f = fopen("image.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image.data(), 1, image.size(), f);
    fclose(f);
    printf("Wrote image.ppm (%dx%d, depth %d)\n", W, H, depth);
    return 0;
}

