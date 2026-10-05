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
    for (int i = d_offsets[u]; i < d_offsets[u + 1]; ++i) {
        int new_dist = dist[u] + d_weights[i];
        int v = d_neighs[i];
        if (new_dist < dist[v]) {
           if (atomicMin(&dist[v], new_dist) > new_dist) {
                *updated = true;
            }
        }
    }   
}











// ============================================================================
// SINGLE SOURCE DELTA-STEPPING DRIVER
// ============================================================================

void naive_approach(
    int source, 
    int* d_tent, 
    int* d_offsets,
    int* d_neighs,
    int* d_weights,
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
    }
    cudaFree(updated);
}

void run_delta_stepping_single_source(
    /*




    */
)
{
    /*
    for adaptive delta, print from within this function. That would be easy.


    */
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

    int *d_offsets = nullptr, *d_neighs = nullptr, *d_weights = nullptr, *d_tent = nullptr;
    cudaMalloc(&d_offsets, (N + 1) * sizeof(int));
    cudaMalloc(&d_neighs, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(int));

    cudaMemcpy(d_offsets, offsets, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighs, neighs, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, E * sizeof(int), cudaMemcpyHostToDevice);

    int *h_tent = new int[ N ] { 0 }; // sssp distance arry. tent means tentative distance
    cudaMalloc(&d_tent, N * sizeof(int));


    /*
    ToDo








    */

    for (int i = 0; i < S_count; ++i)
    {
        int source = sources[i];

        outfile << source << "\n";

        naive_approach(
            source,
            d_tent,
            d_offsets,     
            d_neighs,
            d_weights,
            N,
            E
        );

        // run_delta_stepping_single_source(
        //     /*
        //     ToDo
            





        //     */
        // );

        cudaMemcpy(h_tent, d_tent, N * sizeof(int), cudaMemcpyDeviceToHost);

        for (int v = 0; v < N; ++v)
        {
            outfile << h_tent[v] << "\n";
        }

        for (int v = 0; v < N; ++v)
        {
            cout << h_tent[v] << "\n";
        }
    }

    cudaFree(d_offsets);
    cudaFree(d_neighs);
    cudaFree(d_weights);
    cudaFree(d_tent);

    delete[] h_tent;
    delete[] offsets;
    delete[] neighs;
    delete[] weights;

    outfile.close();
    return 0;
}