#include <cuda_runtime.h>
#include <stdio.h>

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
#define  SIZE 55574528

//#define  SIZE 372244489

#define  MIN_WORK 262144
#define IRR_POLY 0xbfe6b8a5bf378d83

//context for the chunker
#define D_ 512
#define D_DASH 256
#define MIN_SIZE 32768
#define MAX_SIZE 65536

// printf() is only supported
// for devices of compute capability 2.0 and higher
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

void printChunkData_host(int start, int end) {
	printf("Chunk defined:[%d --- %d] [%d]\n", start, end, end - start);
}

void printResults(int* results) {
	for (int var = 0; var < 2; ++var) {
		printf("Chunk defined: %d\n", results[var]);

	}

	//printf("Chunks on CPU: %d , Avg size: %d\n", cnt, size / cnt);
}

__device__ int getTotalThreads() {
	return gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;

}

__device__ int getThrID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

int __device__ __host__ getSizeOfBPArray(int dataLn, int minThreshold) {
	return (dataLn % minThreshold == 0) ? dataLn / minThreshold : (dataLn / minThreshold) + 1;
}

__device__ void getThreadBounds(threadBounds* bounds, int dataLn, int minChunkSize, int workingThreads, int thrID, int workPerThr, int sizeOfBPArray,
		int maxBPsPerThread) {

	bounds->start = thrID * workPerThr;

	//ACCOUTN FOR ANY LEFTOVER DATA THAT CANNOT BE DISTRIBUTED ;)
	bounds->end = (thrID == workingThreads - 1) ? bounds->end = dataLn : bounds->start + workPerThr;

	bounds->BPwritePosition = maxBPsPerThread * thrID;
}

int getNumNeededThreads(int dataLn, int minWorkPerThread) {
	//int minWorkPerThread = 262144;

	//return (dataLn % minSizePerThread != 0) ? (dataLn / minSizePerThread + 1) : dataLn / minSizePerThread;
	return (dataLn <= minWorkPerThread) ? 1 : dataLn / minWorkPerThread;
}

int* allocateBPResultsArray(int sizeOfInput, int minThreshold) {
	int sizeOfBpArray = getSizeOfBPArray(sizeOfInput, minThreshold);
	int* breakpoints;
	breakpoints = (int*) (sizeOfBpArray * sizeof(int));
	return breakpoints;
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

__global__ void RabinExample(rabinData* deviceRabin, BYTE* data, int from, int to, int* results, int threadsUsed, int sizeOfBPArray, int BpsPerThread) {

	int thrID = getThrID();

	if (thrID < threadsUsed) {
//		/printf("%d,%d -----> %d [%d] \n", thrID, dataBounds.start, dataBounds.end, dataBounds.BPwritePosition);

		threadBounds dataBounds;
		/*		dataBounds.BPwritePosition = 1;
		 dataBounds.start = 1;
		 dataBounds.end = 2;*/

		getThreadBounds(&dataBounds, SIZE, MIN_SIZE, threadsUsed, thrID, MIN_WORK, sizeOfBPArray, BpsPerThread);

		//defining contex for the chunker
		chunkingContext ctx;
		ctx.D = D_;
		ctx.Ddash = D_DASH;
		ctx.minThr = MIN_SIZE;
		ctx.maxThr = MAX_SIZE;

		chunkData(deviceRabin, data, dataBounds, ctx, results);
	}
}

/**
 * main function that just launced the kernel and takes care of book kepping
 */

int main() {

	int size = SIZE;
	int sizeOfBParray = getSizeOfBPArray(size, MIN_SIZE);

	printf("size of BP array %d\n", sizeOfBParray);

	BYTE* data = (BYTE*) malloc(sizeof(BYTE) * SIZE);
	int* resultingBreakpoints = (int*) malloc(sizeof(int) * sizeOfBParray);

	for (int var = 0; var < sizeOfBParray; ++var) {
		resultingBreakpoints[var] = 0;
	}

	srand(2);
	for (int var = 0; var < SIZE; ++var) {
		data[var] = (BYTE) rand() % 256;
	}

// host and device data for rabin window context
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);
	rabinData hostData;
	rabinData* deviceData;

// we first init the window on the host.
	initWindow(&hostData, IRR_POLY);

//allocate device memory for the rabin context data

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &deviceData, sizeof(rabinData)));
//copy the data to device
	CUDA_CHECK_RETURN(cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));

// allocate space for the data that we need to chunk and copy it to the device
	BYTE* dataToFingerprint_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &dataToFingerprint_d, sizeof(BYTE) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(dataToFingerprint_d, data, sizeof(BYTE) * size, cudaMemcpyHostToDevice));

//now we allocate some space for the results
	int* resultingBreakpoints_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &resultingBreakpoints_d, sizeof(int) * sizeOfBParray));
	CUDA_CHECK_RETURN(cudaMemcpy(resultingBreakpoints_d, resultingBreakpoints, sizeof(int) * sizeOfBParray, cudaMemcpyHostToDevice));

// call the kernel that will create the fingerprints

	int threadsNeeded = getNumNeededThreads(SIZE, MIN_WORK);

	int blocksize = 160;

	int numBlocks = threadsNeeded / blocksize;
	if (threadsNeeded % blocksize) {
		++numBlocks;
	}

	//printf("Threads per block: %d, Blocks: %d, Total threads: %d, Active threads: %d\n", blocksize, numBlocks, numBlocks * blocksize, threadsNeeded);
	int bpsPerThread = round(((double) (sizeOfBParray)) / threadsNeeded);
	//printf("BP per thread %d\n", sizeOfBParray);

	RabinExample<<<numBlocks, blocksize>>>(deviceData, dataToFingerprint_d, 0, size, resultingBreakpoints_d, threadsNeeded, sizeOfBParray, bpsPerThread);

//copy back into our supplied data
	CUDA_CHECK_RETURN(cudaMemcpy(resultingBreakpoints, resultingBreakpoints_d, sizeof(int) * sizeOfBParray, cudaMemcpyDeviceToHost));

// free all the memory alocated on the cardgetThreadBounds
	CUDA_CHECK_RETURN(cudaFree(deviceData));
	CUDA_CHECK_RETURN(cudaFree(dataToFingerprint_d));
	CUDA_CHECK_RETURN(cudaFree(resultingBreakpoints_d));

	//printResults(resultingBreakpoints);

	free(data);
	free(resultingBreakpoints);

	return 0;
}

