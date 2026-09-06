#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<sys/time.h>
#include<cuda.h>
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
    cudaDeviceSynchronize();  // force the print to flush before continuing
    cudaFree(d_label);
}

__global__ void transpose(int* d_mat, int* t_mat) { //launch with <<<r, c>>> 
	//this is probably buggy
	int i = blockIdx.x, j = threadIdx.x, D = gridDim.x;
	int idx1 = i*D + j;
	int idx2 = j*D + i;
	t_mat[idx2] = d_mat[idx1];
}

__global__ void multiply(int* mat1, int* mat2, int l, int* res) { //launch with <<<p, r>>>
	int i = blockIdx.x, j = threadIdx.x;
	int r = gridDim.x, c = blockDim.x;
	int idx3 = i * c + j;
	res[idx3] = 0;
	for (int x = 0; x < l; x++) {
		int idx1 = i * l + x; 
		int idx2 = x * c + j;
		res[idx3] += mat1[idx1]*mat2[idx2];
	}
}

__global__ void add(int* mat1, int* mat2) { //we'll do this in-place, and just add everything to mat1
	int idx = blockIdx.x * gridDim.x + threadIdx.x;
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
	int *t_matA, *t_matD, *d_matrixTemp;
	cudaMalloc(&t_matA, q * p * sizeof(int));
	cudaMalloc(&t_matD, r * q * sizeof(int));
	cudaMalloc(&d_matrixTemp, p * r * sizeof(int));
	
	debugPrint("A", d_matrixA, p, q);
	transpose<<<q, p>>>(d_matrixA, t_matA);
	debugPrint("AT", t_matA, q, p);

	debugPrint("D", d_matrixD, r, q);
	transpose<<<r, q>>>(d_matrixD, t_matD);
	debugPrint("DT", t_matD, q, r);
	
	// debugPrint("B", d_matrixB, q, r);
	// debugPrint("C", d_matrixC, p, q);
	multiply<<<p, r>>>(t_matA, d_matrixB, q, d_matrixE);
	// debugPrint("E = ATB", d_matrixE, p, r);

	multiply<<<p, r>>>(d_matrixC, t_matD, q, d_matrixTemp);
	// debugPrint("Temp = CDT", d_matrixTemp, p, r);

	add<<<p, r>>>(d_matrixE, d_matrixTemp);
	// debugPrint("Final = E + Temp", d_matrixE, p, r);


	cudaDeviceSynchronize();
	cudaFree(t_matA);
	cudaFree(t_matD);
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
