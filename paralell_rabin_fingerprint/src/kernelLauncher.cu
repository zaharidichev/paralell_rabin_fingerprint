#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>

/*
 * just testing wether we got out inclusion guards right ;)
 */
#include "rabin_fingerprint/Chunker.h"
#include "rabin_fingerprint/RabinFingerprint.h"
#include "rabin_fingerprint/RabinData.h"

#include "data_structures/Buffer.h"
#include "data_structures/Polynomial_128.h"

#include "math/BitOps.h"
#include "math/PolyMath.h"

//typedef uint64_t POLY_64;
//typedef unsigned char BYTE;
#define  SIZE 104857


 void printChunkData_host(long start, long end) {
	printf("Chunk defined:[%d --- %d] [%d]\n", start, end, end - start);
}


void printResults(long* results) {

	long cnt = 1;
	long size = 0;
	while (results[cnt] != -1) {
		printChunkData_host(results[cnt-1], results[cnt]);
		size = size + (results[cnt] - results[cnt-1]);
		cnt++;
	}
	printf("Chunks on CPU: %d , Avg size: %d\n", cnt,
			size / cnt);
}

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__global__ void RabinExample(rabinData* deviceRabin, BYTE* data, long from,
		long to, long* results) {

	chunkData(deviceRabin, data, from, to, 512, 256, 460, 2800, results);

}

/**
 * main function that just launced the kernel and takes care of book kepping
 */

int main() {
	int size = SIZE;

	BYTE* data = (BYTE*) malloc(sizeof(BYTE) * SIZE);
	long* breakPoints = (long*) malloc(sizeof(long) * SIZE);
	long* resultingBreakpoints = (long*) malloc(sizeof(long) * ((SIZE/460) + 1));

	srand(2);
	for (int var = 0; var < SIZE; ++var) {
		data[var] = (BYTE) rand() % 256;
	}

	/*
	 * host and device data for rabin window context
	 */
	rabinData hostData;
	rabinData* deviceData;

	// we first init the window on the host.
	initWindow(&hostData, 0xbfe6b8a5bf378d83);

	//allocate device memory for the rabin context data
	cudaMalloc((void**) &deviceData, sizeof(rabinData));

	//copy the data to device
	CUDA_CHECK_RETURN(
			cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));

	// allocate space for the data that we need to chunk and copy it to the device
	BYTE* dataToFingerprint_d;
	cudaMalloc((void**) &dataToFingerprint_d, sizeof(BYTE) * size);
	CUDA_CHECK_RETURN(
			cudaMemcpy(dataToFingerprint_d, data, sizeof(BYTE) * size, cudaMemcpyHostToDevice));

	//now we allocate some space for the results
	long* resultingBreakpoints_d;
	cudaMalloc((void**) &resultingBreakpoints_d, sizeof(long) *   ((SIZE/460) + 1));

	// call the kernel that will create the fingerprints

	cudaPrintfInit();
	RabinExample<<<1, 1>>>(deviceData, dataToFingerprint_d, 0, size, resultingBreakpoints_d);
	cudaPrintfDisplay(stdout, true);
	cudaPrintfEnd();

	//copy back into our supplied data
	CUDA_CHECK_RETURN(
			cudaMemcpy(resultingBreakpoints, resultingBreakpoints_d, sizeof(long) *  ((SIZE/460) + 1), cudaMemcpyDeviceToHost));

	// free all the memory alocated on the card
	CUDA_CHECK_RETURN(cudaFree(deviceData));
	CUDA_CHECK_RETURN(cudaFree(dataToFingerprint_d));
	CUDA_CHECK_RETURN(cudaFree(resultingBreakpoints_d));

	printResults(resultingBreakpoints);


	free(data);
	free(breakPoints);
	free(resultingBreakpoints);

	return 0;
}

