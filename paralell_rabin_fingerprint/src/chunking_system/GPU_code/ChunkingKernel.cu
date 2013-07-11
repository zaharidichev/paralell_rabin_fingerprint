/**
 * ChunkingKernel.cu
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef CHUNKINGKERNEL_CU_
#define CHUNKINGKERNEL_CU_

#include "../../rabin_fingerprint/Chunker.h"
#include  "cuda_runtime.h"
#include "ResourceManagement.h"
#include "KernelStarter.h"
#include "BitFieldArray.h"
#include <iostream>
#include <fstream>      // std::ifstream
#include "../../etc/helpers/Macros.h"
#include "hashing/sha1_kernel.cu"
#include "openssl/sha.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__device__ int getThrID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ void getThreadBounds(threadBounds* bounds, int dataLn, int threadsUsed, int thrID, int workPerThr) {

	bounds->start = thrID * workPerThr;

	//ACCOUTN FOR ANY LEFTOVER DATA THAT CANNOT BE DISTRIBUTED ;)
	bounds->end = (thrID == threadsUsed - 1) ? bounds->end = dataLn : bounds->start + workPerThr;

}

__global__ void findBreakPointsSegmented(rabinData* deviceRabin, chunkingContext* ctx, BYTE* data, int dataLen, int* results, int threadsUsed, BYTE* hashes) {

	int thrID = getThrID();

	if (thrID < threadsUsed) {

		threadBounds dataBounds;
		getThreadBounds(&dataBounds, dataLen, threadsUsed, thrID, ctx->workPerThread);

		chunkDataWithLimits(deviceRabin, data, dataBounds, ctx, results, threadsUsed, hashes);
		//chunkDataWithLimits(deviceRabin, data, dataBounds, ctx, results);

	}

}

__global__ void findBreakPointsFreeMode(rabinData* deviceRabin, BYTE* data, int dataLen, bitFieldArray results, int threadsUsed, int workPerThread,
		int divisor) {

	int thrID = getThrID();

	if (thrID < threadsUsed) {

		threadBounds dataBounds;

		getThreadBounds(&dataBounds, dataLen, threadsUsed, thrID, workPerThread);

		chunkDataFreeMode(deviceRabin, data, dataBounds, divisor, results, threadsUsed);
	}
}

void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results, int threadsUsed,
		int workPerThread, int D) {
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);

	findBreakPointsFreeMode<<<numBlocks, blocksSize>>>(deviceRabin, deviceData, dataLen, results, threadsUsed, workPerThread, D);

	gpuErrchk(cudaGetLastError());

	cudaThreadSynchronize();
}

void startSegmentedChunkingAndHashingKernel(size_t blocksSize, size_t numBlocks, rabinData* rabinData_d, chunkingContext* ctx_d, BYTE* dataToChunk_d,
		size_t sizeOfData, size_t activeThreads, BYTE* hashed_d, int* results_d) {


	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);

	findBreakPointsSegmented<<<numBlocks, blocksSize>>>(rabinData_d,ctx_d,dataToChunk_d,sizeOfData,results_d,activeThreads,hashed_d);

	gpuErrchk(cudaGetLastError());

	cudaThreadSynchronize();
}

int __host__ getSizeOfBPArray(int dataLn, int minThreshold) {
	return (dataLn % minThreshold == 0) ? dataLn / minThreshold : (dataLn / minThreshold) + 1;
}

int f() {

	std::ifstream infile("/home/zahari/Desktop/data.txt", std::ofstream::binary);

	int sizeOfData = 536870912;
	int minSize = 32768;
	int maxSize = 131072;

	int sizeOfBParray = getSizeOfBPArray(sizeOfData, minSize);
	unsigned char* data = (unsigned char*) malloc(sizeOfData);
	infile.read((char*) data, sizeOfData);

	// host and device data for rabin window context
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);
	rabinData hostData;
	rabinData* deviceData;

	// we first init the window on the host.
	initWindow(&hostData, 0xbfe6b8a5bf378d83);

	//allocate device memory for the rabin context data

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &deviceData, sizeof(rabinData)));
	//copy the data to device
	CUDA_CHECK_RETURN(cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));

	// allocate space for the data that we need to chunk and copy it to the device
	BYTE* dataToFingerprint_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &dataToFingerprint_d, sizeof(BYTE) * sizeOfData));
	CUDA_CHECK_RETURN(cudaMemcpy(dataToFingerprint_d, data, sizeof(BYTE) * sizeOfData, cudaMemcpyHostToDevice));

	//now we allocate some space for the results
	int* resultingBreakpoints_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &resultingBreakpoints_d, sizeof(int) * sizeOfBParray));
	CUDA_CHECK_RETURN(cudaMemset(resultingBreakpoints_d, 0, sizeof(int) * sizeOfBParray));

	int threadsNeeded = getNumNeededThreads(sizeOfData, 262144);

	int blocksize = 160;

	int numBlocks = threadsNeeded / blocksize;
	if (threadsNeeded % blocksize) {
		++numBlocks;
	}

	BYTE* hashes = (BYTE*) (malloc(sizeof(BYTE) * sizeOfBParray * 20));

	BYTE* hashes_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &hashes_d, sizeof(BYTE) * sizeOfBParray * 20));

	int bpsPerThread = round(((double) (sizeOfBParray)) / threadsNeeded);

	chunkingContext ctx;

	ctx.BpreakpointsPerThread = bpsPerThread;
	ctx.D = 512;
	ctx.Ddash = 256;
	ctx.maxThr = maxSize;
	ctx.minThr = minSize;
	ctx.sizeOfBreakpointsArray = sizeOfBParray;
	ctx.workPerThread = 262144;

	chunkingContext* ctx_d;

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &ctx_d, sizeof(chunkingContext)));
	CUDA_CHECK_RETURN(cudaMemcpy(ctx_d, &ctx, sizeof(chunkingContext), cudaMemcpyHostToDevice));

	findBreakPointsSegmented<<<numBlocks, blocksize>>>(deviceData, ctx_d, dataToFingerprint_d, sizeOfData, resultingBreakpoints_d, threadsNeeded, hashes_d);

	int* resultingBreakpoints = (int*) malloc(sizeof(int) * sizeOfBParray);

	//copy back into our supplied data
	CUDA_CHECK_RETURN(cudaMemcpy(resultingBreakpoints, resultingBreakpoints_d, sizeof(int) * sizeOfBParray, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(hashes, hashes_d, sizeof(BYTE) * sizeOfBParray * 20, cudaMemcpyDeviceToHost));

	// free all the memory alocated on the cardgetThreadBounds
	CUDA_CHECK_RETURN(cudaFree(deviceData));
	CUDA_CHECK_RETURN(cudaFree(dataToFingerprint_d));
	CUDA_CHECK_RETURN(cudaFree(resultingBreakpoints_d));

	for (int var = 0; var < sizeOfBParray; ++var) {

		printf("%d\n", resultingBreakpoints[var]);

	}

	for (int i = 0; i < sizeOfBParray; ++i) {

		for (int var = 0; var < 20; ++var) {
			printf("%02x", hashes[i * 20 + var]);

		}
		printf("\n");

	}

	free(data);
	free(resultingBreakpoints);

	unsigned char* buffer = (unsigned char*) malloc(33135);
	std::ifstream infile2("/home/zahari/Desktop/data.txt", std::ofstream::binary);
	infile.seekg(536775699);

	infile.read((char*) buffer, 33135);
	BYTE* digest = (BYTE*) malloc(20);

	SHA1(buffer, 33135, digest);
	printf("----------------------------------\n");

	for (int var = 0; var < 20; ++var) {
		printf("%02x", digest[var]);

	}

}

#endif /* CHUNKINGKERNEL_CU_ */
