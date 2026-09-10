#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<sys/time.h>
#include<cuda.h>
#define BLOCK 1024
using namespace std;

__global__ void print(const char* name, int* mat, int r, int c) {
	printf("%s\n", name);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			int idx = i * c + j;
			printf("%4d", mat[idx]);
		}
		printf("\n");
	}
}

void debugPrint(const char* label, int* d_mat, int r, int c) {
    char *d_label;
    cudaMalloc(&d_label, strlen(label) + 1);
    cudaMemcpy(d_label, label, strlen(label) + 1, cudaMemcpyHostToDevice);
    print<<<1, 1>>>(d_label, d_mat, r, c);
    cudaDeviceSynchronize();  
    cudaFree(d_label);
}


__global__ void transpose(int* d_mat, int r, int c, int* t_mat) { //we'll launch this with <<ceil(r/32)*ceil(c/32), BLOCK>>
	__shared__ int tile[1024];
	//threadIdx.x is from 0 to 1023
	int tpr = (c + 31)/32; //tiles per row
	int bl_r = threadIdx.x / 32, bl_c = threadIdx.x % 32;
	int g_r = blockIdx.x / tpr, g_c = blockIdx.x % tpr; 
	//these are indices within the 32*32 shared memory
	int shared = bl_r * 32 + bl_c;

	int matr = 32*g_r + bl_r, matc = 32*g_c + bl_c;
	if (matr < r && matc < c) tile[shared] = d_mat[matr * c + matc];
	
	__syncthreads(); //fills the tile
	//tile now has a 32*32 chunk of our matrix
	int tp_r = g_c, tp_c = g_r;
	shared = bl_c * 32 + bl_r;
	int t_matr = 32*tp_r + bl_r, t_matc = 32*tp_c + bl_c;
	if (t_matr < c && t_matc < r) t_mat[t_matr * r + t_matc] = tile[shared];
}

// __global__ void multiply(int* mat1, int* mat2, int p, int q, int r, int* res) { //call this with ceil(p/16)*ceil(q/16)*ceil(r/16) blocks
// 	__shared__ int tiles[512]; //16*16 tile per matrix
// 	// printf("Entered multiply\n");
// 	int i = blockIdx.x, j = blockIdx.y, k = blockIdx.z; //launching using dim3 instead
// 	int bl_r = threadIdx.x / 16, bl_c = threadIdx.x % 16;
// 	if (threadIdx.x != blockDim.x - 1) { //we'll pass in 1 extra just for the write-back
// 		// matrix 1
//         int act_r1 = 16 * i + bl_r, act_c1 = 16 * j + bl_c;
//         int idx1 = 16 * bl_r + bl_c;
//         tiles[idx1] = (act_r1 < p && act_c1 < q) ? mat1[act_r1 * q + act_c1] : 0;

//         // matrix 2
//         int act_r2 = 16 * j + bl_r, act_c2 = 16 * k + bl_c;
//         int idx2 = 256 + 16 * bl_r + bl_c;
//         tiles[idx2] = (act_r2 < q && act_c2 < r) ? mat2[act_r2 * r + act_c2] : 0;
// 	}
// 	__syncthreads();
// 	if (threadIdx.x == blockDim.x - 1) {
// 		for (int x = 0; x < 16; x++) {
// 			for (int y = 0; y < 16; y++) {
// 				int res_r = 16 * i + x, res_c = 16 * k + y; // this is the target location
// 				if (res_r < p && res_c < r) {
// 					int acc = 0;
// 					for (int z = 0; z < 16; z++) {
// 					    int idx1 = x * 16 + z;        
// 					    int idx2 = 256 + z * 16 + y;  
// 					    acc += tiles[idx1] * tiles[idx2];
// 					}
// 					atomicAdd(&res[res_r * r + res_c], acc);
// 				}
// 			}
// 		}
// 	}
// }

//we'll multiply using A-B in A-BT form
__global__ void multiply(int* mat1, int* mat2, int p, int q, int r, int* res) { //call this with ceil(p/32)*ceil(q/32)*ceil(r/32) blocks
	__shared__ int tiles[2048]; //32*16 tile per matrix
	// printf("Entered multiply\n");
	int i = blockIdx.x, j = blockIdx.y, k = blockIdx.z; //launching using dim3 instead
	int bl_r = threadIdx.x / 32, bl_c = threadIdx.x % 32;
	
	// matrix 1
    int act_r1 = 32 * i + bl_r, act_c1 = 32 * j + bl_c;
    int idx1 = 32 * bl_r + bl_c;
    tiles[idx1] = (act_r1 < p && act_c1 < q) ? mat1[act_r1 * q + act_c1] : 0;

    // matrix 2
    int act_r2 = 32 * k + bl_r, act_c2 = 32 * j + bl_c;
    int idx2 = 1024 + 32 * bl_r + bl_c;
    tiles[idx2] = (act_r2 < r && act_c2 < q) ? mat2[act_r2 * q + act_c2] : 0; //we will pass the transpose here
	
	__syncthreads();
	// if (threadIdx.x == 0) {
	// 	for (int x = 0; x < 32; x++) {
	// 		for (int y = 0; y < 32; y++) {
				int res_r = 32 * i + bl_r, res_c = 32 * k + bl_c;
				if (res_r < p && res_c < r) {
					int acc = 0;
					for (int z = 0; z < 32; z++) {
					    int idx1 = bl_r * 32 + z;        
					    int idx2 = 1024 + bl_c * 32 + z;  
					    acc += tiles[idx1] * tiles[idx2];
					}
					atomicAdd(&res[res_r * r + res_c], acc);
	// 			}
	// 		}
	// 	}
	// }
}

__global__ void add(int* mat1, int* mat2) { //we'll do this in-place, and just add everything to mat1
	//not using shared memory, memory coalescing is enough (I think)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	mat1[idx] += mat2[idx];
}

// function to compute the output matrix
void compute(int p, int q, int r, int *h_matrixA, int *h_matrixB,
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;

	// allocate memory...
	cudaMalloc(&d_matrixA, q * p * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	int *t_matA, *t_matB, *d_matrixTemp;
	cudaMalloc(&t_matA, q * p * sizeof(int));
	cudaMalloc(&t_matB, q * r * sizeof(int));
	cudaMalloc(&d_matrixTemp, p * r * sizeof(int));
	
	int row = (q + 31)/32, col = (p + 31)/32;
	transpose<<<row*col, BLOCK>>>(d_matrixA, q, p, t_matA);

	row = (r + 31)/32, col = (q + 31)/32;
	transpose<<<row*col, BLOCK>>>(d_matrixB, q, r, t_matB);
	
	cudaMemset(d_matrixE, 0, p*r*sizeof(int));

	int x = (p + 31)/32, y = (q + 31)/32, z = (r + 31)/32;
	multiply<<<dim3(x, y, z), BLOCK>>>(t_matA, t_matB, p, q, r, d_matrixE);
	
	cudaMemset(d_matrixTemp, 0, p*r*sizeof(int));
	multiply<<<dim3(x, y, z), BLOCK>>>(d_matrixC, d_matrixD, p, q, r, d_matrixTemp);

	add<<<p, r>>>(d_matrixE, d_matrixTemp);
	cudaDeviceSynchronize();

	cudaDeviceSynchronize();
	cudaFree(t_matA);
	cudaFree(t_matB);
	cudaFree(d_matrixTemp);
	
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}



int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n");
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(q * p * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, q, p);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	compute(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
