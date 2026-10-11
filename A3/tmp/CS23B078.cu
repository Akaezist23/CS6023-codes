#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/remove.h>
#include <thrust/fill.h>

#define INF 2147483647 // simply INT_MAX
#define BLOCK_SIZE 256 // you can change it.
#define NONE 0
#define NEAR 1
#define FAR 2

// ============================================================================
// CUDA KERNELS
// ============================================================================
__global__ void bellman_ford(
    int* dist,
    int* d_offsets,
    int* d_neighs,
    int* d_weights,
    int N,
    int E,
    bool* updated
) 
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N || dist[u] == INF) return;
    for (int i = d_offsets[u]; i < d_offsets[u + 1]; i++) {
        int new_dist = dist[u] + d_weights[i];
        int v = d_neighs[i];
        if (new_dist < dist[v]) {
           if (atomicMin(&dist[v], new_dist) > new_dist) {
                *updated = true;
            }
        }
    }   
}

__global__ void light_phase(
    int* dist, 
    int* d_offsets,
    int* d_neighs,
    int* d_weights,
    int* d_state,
    int* d_bucket,
    int* d_near_in,
    int* d_near_out,
    int* d_far,
    int* d_set,
    int* near_in,
    int* near_out,
    int* set_cnt,
    int* far_idx,
    int* curr_bucket,
    int* cutoff,
    int* delta
) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *near_in) return;
    int u = d_near_in[idx];
    atomicExch(&d_state[u], NONE);
    __threadfence(); // makes sure ur change is visible to other threads

    if (atomicExch(&d_bucket[u], *curr_bucket) != *curr_bucket) { 
        __threadfence(); // check why this thing is here
        d_set[atomicAdd(set_cnt, 1)] = u;
    }
    int du = atomicAdd(&dist[u], 0); // to prevent cached reads
    for (int i = d_offsets[u]; i < d_offsets[u + 1]; i++) {
        int w = d_weights[i];
        if (w > *delta) continue;
        int new_dist = du + w;
        int v = d_neighs[i];
        if (atomicMin(&dist[v], new_dist) > new_dist) {
            if (new_dist <= *cutoff) {
                if (atomicExch(&d_state[v], NEAR) != NEAR) { //check this part, might be incorrect
                    d_near_out[atomicAdd(near_out, 1)] = v;
                }
            }
            else if (atomicCAS(&d_state[v], NONE, FAR) == NONE) {
                d_far[atomicAdd(far_idx, 1)] = v;
            }
        }
    }
}

__global__ void heavy_phase(
    int* dist,
    int* d_offsets,
    int* d_neighs,
    int* d_weights,
    int* d_state,
    int* d_set,
    int* d_far,
    int* far_idx, 
    int* set_cnt,
    int* delta
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *set_cnt) return;
    int u = d_set[idx];
    for (int i = d_offsets[u]; i < d_offsets[u + 1]; i++) {
        int w = d_weights[i];
        if (w <= *delta) continue;
        int new_dist = dist[u] + w; 
        int v = d_neighs[i];
        if (atomicMin(&dist[v], new_dist) > new_dist) {
            if (atomicCAS(&d_state[v], NONE, FAR) == NONE) {
                d_far[atomicAdd(far_idx, 1)] = v;
            }
        }
    }
}

__global__ void mark_near(const int* near, int* state, int cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cnt) state[near[i]] = NEAR;
}

struct LessThan {
    int* dist;
    int val;
    __device__ bool operator()(int v) {
        return dist[v] <= val;
    }
};

struct GetDist {
    int* dist;
    __device__ int operator()(int v) {
        return dist[v];
    }
};

// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void naive_approach(
    int source, 
    int* d_tent, 
    int* d_offsets,
    int* d_neighs,
    int* d_weights,
    int* d_settled,
    int N,
    int E
) 
{
    thrust::device_ptr<int> dist(d_tent);
    thrust::fill(dist, dist + N, INF);
    dist[source] = 0;
    bool* updated;
    cudaMallocManaged(&updated, sizeof(bool));
    *updated = true;

    while (*updated) {
        *updated = false;
        bellman_ford<<<ceil((double)N / BLOCK_SIZE), BLOCK_SIZE>>>(dist.get(), d_offsets, d_neighs, d_weights, N, E, updated);
        cudaDeviceSynchronize();
    }
    cudaFree(updated);
}

void run_delta_stepping_single_source(
    int source, 
    int* d_tent,
    int* d_offsets,
    int* d_neighs,
    int* d_weights,
    int* d_far,
    int* d_near_in,
    int* d_near_out,
    int* d_state,
    int* d_bucket,
    int* d_set,
    int* d_keys, 
    int N,
    int E,
    int delta_mode,
    int _K,
    std::ofstream& outfile
)
{
    /*
    for adaptive delta, print from within this function. That would be easy.
    */

    thrust::device_ptr<int> dist(d_tent);
    thrust::fill(dist, dist + N, INF);
    dist[source] = 0;

    thrust::device_ptr<int> state(d_state);
    thrust::fill(state, state + N, NONE); 
    state[source] = NEAR;

    thrust::device_ptr<int> bucket(d_bucket);
    thrust::fill(bucket, bucket + N, -1);

    thrust::device_ptr<int> near_i(d_near_in);
    thrust::device_ptr<int> near_o(d_near_out);
    thrust::device_ptr<int> far(d_far);
    thrust::device_ptr<int> set(d_set);
    thrust::device_ptr<int> keys(d_keys);

    bool is_static = false;

    int* near_in; 
    int* near_out;
    int* curr_bucket;
    int* cutoff;
    int* prev_cutoff;
    int* delta;
    int* K;
    int* set_cnt;
    int* far_idx;
    cudaMallocManaged(&near_in, sizeof(int));
    cudaMallocManaged(&near_out, sizeof(int));
    cudaMallocManaged(&curr_bucket, sizeof(int));
    cudaMallocManaged(&cutoff, sizeof(int));
    cudaMallocManaged(&prev_cutoff, sizeof(int));
    cudaMallocManaged(&delta, sizeof(int));
    cudaMallocManaged(&K, sizeof(int));
    cudaMallocManaged(&set_cnt, sizeof(int));
    cudaMallocManaged(&far_idx, sizeof(int));

    *K = _K;
    *curr_bucket = 0;
    *prev_cutoff = 0;
    if (delta_mode >= 0) { // STATIC
        is_static = true;
        *cutoff = delta_mode;
        *delta = delta_mode;
    } 
    else { // ADAPTiVE
        *cutoff = 0;
        *delta = 1; 
    }

    near_i[0] = source;
    *near_in = 1;
    *near_out = 0;
    *far_idx = 0;
    while (true) {
        *set_cnt = 0;
        // light phase
        while (*near_in > 0) {
            int count = *near_in;
            int num_blocks = ceil((double)count / BLOCK_SIZE);
            light_phase<<<num_blocks, BLOCK_SIZE>>>(
                d_tent,
                d_offsets,
                d_neighs,
                d_weights,
                d_state,
                d_bucket,
                near_i.get(),
                near_o.get(),
                d_far,
                d_set,
                near_in,
                near_out,
                set_cnt,
                far_idx,
                curr_bucket,
                cutoff,
                delta
            );
            cudaDeviceSynchronize();
            swap(near_i, near_o);
            *near_in = *near_out;
            *near_out = 0;
        }
        // heavy phase
        int count = *set_cnt;
        if (count > 0) {
            int num_blocks = ceil((double)count / BLOCK_SIZE);
            heavy_phase<<<num_blocks, BLOCK_SIZE>>>(
                dist.get(),
                d_offsets,
                d_neighs,
                d_weights,
                d_state,
                d_set,
                d_far,
                far_idx,
                set_cnt,
                delta
            );
            cudaDeviceSynchronize();
        }
        
        // refill phase
        *prev_cutoff = *cutoff;
        int prev = *prev_cutoff;
        auto end = thrust::remove_if(far, far + *far_idx, LessThan{dist.get(), prev});
        *far_idx = end - far;
        if (*far_idx == 0) break;

        if (is_static) {
            int Dmin = thrust::transform_reduce(far, far + *far_idx, GetDist{dist.get()}, INF, thrust::minimum<int>());
            *cutoff = *delta + Dmin;  
            int cf = *cutoff;
            auto n_end = thrust::copy_if(far, far + *far_idx, near_i, LessThan{dist.get(), cf});
            *near_in = n_end - near_i;
            auto f_end = thrust::remove_if(far, far + *far_idx, LessThan{dist.get(), cf});
            *far_idx = f_end - far;
        }

        else {
            thrust::transform(far, far + *far_idx, keys, GetDist{dist.get()});
            thrust::sort_by_key(keys, keys + *far_idx, far);
            int Dmin = keys[0];
            int i = min(*K - 1, *far_idx - 1);
            *cutoff = keys[i];
            int det = *cutoff - Dmin;
            outfile << det << "\n";
            *delta = max(1, det);
            int cnt = thrust::upper_bound(keys, keys + *far_idx, *cutoff) - keys; //check this thing's correctness later
            int rem = *far_idx - cnt;
            thrust::copy(far, far + cnt, near_i);
            thrust::copy(far + cnt, far + *far_idx, near_o); //reusing near_o as a temp buffer
            thrust::copy(near_o, near_o + rem, far);
            *near_in = cnt;
            *far_idx = rem;
        }

        mark_near<<<ceil((double)*near_in / BLOCK_SIZE), BLOCK_SIZE>>>(near_i.get(), d_state, *near_in);
        cudaDeviceSynchronize();
        (*curr_bucket)++;
    }

    cudaFree(near_in);
    cudaFree(near_out);
    cudaFree(curr_bucket);
    cudaFree(cutoff);
    cudaFree(prev_cutoff);
    cudaFree(delta);
    cudaFree(K);
    cudaFree(set_cnt);
    cudaFree(far_idx);
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
        return 1;
    }

    std::ifstream infile(argv[1]);
    if (!infile.is_open())
    {
        std::cerr << "Error: Unable to open input file " << argv[1] << "\n";
        return 1;
    }

    std::ofstream outfile(argv[2]);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Unable to open output file " << argv[2] << "\n";
        return 1;
    }

    int delta_mode, K, N, E, S_count;
    infile >> delta_mode >> K;
    infile >> N >> E >> S_count;

    std::vector<int> sources(S_count);
    for (int i = 0; i < S_count; ++i)
    {
        infile >> sources[i];
    }

    int *offsets = new int[ N + 1 ] { 0 };
    for (int i = 0; i <= N; ++i)
    {
        infile >> offsets[i];
    }

    int *neighs = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> neighs[i];
    }

    int *weights = new int[ E ] { 0 };
    for (int i = 0; i < E; ++i)
    {
        infile >> weights[i];
    }

    infile.close();

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr, *d_tent = nullptr, *d_state = nullptr, *d_bucket = nullptr;
    int *d_near_in = nullptr, *d_near_out = nullptr, *d_far = nullptr, *d_set = nullptr, *d_keys = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[ N ] { 0 }; // sssp distance arry. tent means tentative distance
    cudaMalloc(&d_tent, N * sizeof(int));
    cudaMalloc(&d_state, N * sizeof(int));
    cudaMalloc(&d_near_in, N * sizeof(int));
    cudaMalloc(&d_near_out, N * sizeof(int));
    cudaMalloc(&d_far, N * sizeof(int));
    cudaMalloc(&d_bucket, N * sizeof(int));
    cudaMalloc(&d_set, N * sizeof(int));
    cudaMalloc(&d_keys, N * sizeof(int));


    /*
    ToDo








    */

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];

        outfile << source << "\n";

        // naive_approach(
        //     source,
        //     d_tent,
        //     d_offsets,     
        //     d_neighs,
        //     d_weights,
        //     N,
        //     E
        // );

        run_delta_stepping_single_source(
            source,
            d_tent,
            d_offsets,
            d_neighs,
            d_weights,
            d_far,
            d_near_in,
            d_near_out,
            d_state,
            d_bucket,
            d_set,
            d_keys,
            N,
            E,
            delta_mode,
            K,
            outfile
        );

        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }

        // for (int v = 0; v < N; ++v)
        // {
        //     std::cerr << h_tent[v] << "\n";
        // }
    }

    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_tent);
    cudaFree(d_state);
    cudaFree(d_near_in);
    cudaFree(d_near_out);
    cudaFree(d_far);
    cudaFree(d_bucket);
    cudaFree(d_set);
    cudaFree(d_keys);

    delete[] h_tent;
    delete[] offsets;
    delete[] neighs;
    delete[] weights;

    outfile.close();
    return 0;
}