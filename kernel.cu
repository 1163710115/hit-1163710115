# pragma warning (disable:4819)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define ARRAYSIZE 5

#define checkCudaErrors( a ) do { \
    if (cudaSuccess != (a)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);


void printDeviceProp(cudaDeviceProp &prop);
void printVector(const int vector[]);

void inquireGPUInfo() {

	int count;

	cudaGetDeviceCount(&count); 
	if (count == 0) {
		printf("There is no device.\n");
		return;
	} else {
		printf("Device count is %d.\n\n", count);
	}

	// find the device
	int i;
	for (i = 0; i < count; ++i) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
				printDeviceProp(prop);
		}
	}
	printf("\n");
}

void printDeviceProp(cudaDeviceProp &prop)
{
	printf("Device name :\t %s.\n", prop.name);
	printf("Major compute capability: \t %d.\n", prop.major);
	printf("Total global memory: \t %lld bytes.\n", prop.totalGlobalMem);
	printf("Max threads per block: \t %d.\n", prop.maxThreadsPerBlock);
	printf("Total const memory: \t %lld bytes.\n", prop.totalConstMem);
	printf("Shared memory per block: \t %lld bytes.\n", prop.sharedMemPerBlock);
	printf("Registers per block: \t %d.\n", prop.regsPerBlock);
	printf("Max threads per multiprocessors: \t %d.\n", prop.maxThreadsPerMultiProcessor);
	printf("Multiprocessors count: \t %d.\n", prop.multiProcessorCount);
}

__global__ void addKernel(int *c, const int *a, const int *b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void printVector(const int vector[]) {
	int i;
	for (i = 0; i < ARRAYSIZE; i++) {
		if (i != ARRAYSIZE) {
			printf("%d,  ", vector[i]);
		}
		else {
			printf("%d ", vector[i]);
		}
	}
	printf("\n");
}

int main() {
//	inquireGPUInfo();

	const int a[ARRAYSIZE] = { 1, 2, 3, 4, 5 };
	const int b[ARRAYSIZE] = { 10, 20, 30, 40, 50 };
	int c[ARRAYSIZE] = { 0 };

	int *dev_a, *dev_b, *dev_c;
	checkCudaErrors(cudaMalloc((void**)&dev_a, ARRAYSIZE * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_b, ARRAYSIZE * sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dev_c, ARRAYSIZE * sizeof(int)));

	checkCudaErrors(cudaMemcpy(dev_a, a, ARRAYSIZE * sizeof(int)
		, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_b, b, ARRAYSIZE * sizeof(int)
		, cudaMemcpyHostToDevice));

	addKernel <<< 1, ARRAYSIZE >>> (dev_c, dev_a, dev_b);

	checkCudaErrors(cudaMemcpy(c, dev_c, ARRAYSIZE * sizeof(int)
		, cudaMemcpyDeviceToHost));

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	printf("Array 1: \t");
	printVector(a);
	printf("Array 2: \t");
	printVector(b);
	printf("Arrays sum: \t");
	printVector(c);

	getchar();
	return 0;
}