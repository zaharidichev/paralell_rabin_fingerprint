/**
 * ChunkingKernel.cu
 *
 * This file contains the implementation of the functions that are used to start the chunking kernels.
 * Furthermore, it contains the implementation of the chunking kernels themselves. The calls from this
 * file perform the actual computations on the GPU hardware with the help of auxiliary functions such as
 * rabin fingerprint methods SHA-1 hash functions , etc.
 *
 *
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

/**
 *
 * Returns the size of breakpoints that is needed for the segmented breakpoint chunking,
 * given the minimum  threshold. Note that not all of that memory is used, This is a worst
 * case scenario
 *
 * @param dataLn the size of the data
 * @param minThreshold the minimum threshold
 * @return the number of breakpoints needed
 */
int __host__ getSizeOfBPArray(int dataLn, int minThreshold) {
	return (dataLn % minThreshold == 0) ? dataLn / minThreshold : (dataLn / minThreshold) + 1;
}

/**
 * Retrieves the global ID of the current GPU thread
 *
 * @return the id of the thread
 */
__device__ int getThrID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * Determines the bounds of the data that each thread needs to operate on, given its
 * id and the size of the total data to be processed.
 *
 * @param bounds pointer to the bounds structure
 * @param dataLn the total size of the data
 * @param threadsUsed the number of threads that the grid uses
 * @param thrID the id of the current thread
 * @param workPerThr the amount of work per thread
 */
__device__ void getThreadBounds(threadBounds* bounds, int dataLn, int threadsUsed, int thrID, int workPerThr) {

	bounds->start = thrID * workPerThr;

	//ACCOUTN FOR ANY LEFTOVER DATA THAT CANNOT BE DISTRIBUTED ;)
	bounds->end = (thrID == threadsUsed - 1) ? bounds->end = dataLn : bounds->start + workPerThr;

}

/**
 *
 * This is the kernel that performs segmented breakpoint identification and produces
 * chunks that ultimately adhere to the specified minimum and maximum size. In addition,
 * it also produces hashes of the content of those chunks .
 *
 * @param deviceRabin the device rabin data such as push tables and so on
 * @param ctx the context that contains all the settings
 * @param data the data to be chunked
 * @param dataLen the size of the data to be chunked
 * @param results the array that will hold the breakpoints position
 * @param threadsUsed the number of active threads used to perform the oepration
 * @param hashes the array that will hold all the hashes
 */
__global__ void findBreakPointsSegmented(rabinData* deviceRabin, chunkingContext* ctx, BYTE* data, int dataLen, int* results, int threadsUsed, BYTE* hashes) {

	int thrID = getThrID();

	if (thrID < threadsUsed) {

		// grap the threadboudns for this thread
		threadBounds dataBounds;
		getThreadBounds(&dataBounds, dataLen, threadsUsed, thrID, ctx->workPerThread);
		//invoke the actual chunking function
		chunkDataWithLimits(deviceRabin, data, dataBounds, ctx, results, threadsUsed, hashes);

	}

}

/**
 *
 * This kernel just finds raw breakpoints without adhering to minimum and maximum threshold.
 * All of that is left to the host - side code to handle (as well as the hash  computation).
 *
 *
 * @param deviceRabin pointer to the push and pop table for the rabin function
 * @param data pointer to the binary data to be fingerprinted
 * @param dataLen the size of the data
 * @param results pointer to the array that will hold the results
 * @param threadsUsed the actual number of active threads
 * @param workPerThread the amount of work per thread
 * @param divisor the main divisor that is used
 */
__global__ void findBreakPointsFreeMode(rabinData* deviceRabin, BYTE* data, int dataLen, bitFieldArray results, int threadsUsed, int workPerThread,
		int divisor) {

	int thrID = getThrID();

	if (thrID < threadsUsed) {

		threadBounds dataBounds;

		getThreadBounds(&dataBounds, dataLen, threadsUsed, thrID, workPerThread);
		//call the chunking function
		chunkDataFreeMode(deviceRabin, data, dataBounds, divisor, results, threadsUsed);
	}
}

/**
 *
 *	This function is used to start the creation of raw breakpoints on the device via invoking a kernel
 *	launch. The function is compiled as pure C code and can be called from either C++ or C. This particular
 *	kernel just creates breakpoints on the device without performing hash computations. Those are the initial
 *	breakpoints that later on need to be fused in order to fit into the min/max constrains
 *
 * @param blocksSize the number of threads within a block
 * @param numBlocks the number of blocks for the kernel launch
 * @param deviceRabin the device rabin data that was initialized on the device
 * @param deviceData a pointer to the binary data that needs to fingerprinted
 * @param dataLen the size of the data in bytes
 * @param results a pointer to the memory that will hold the resulting raw breakpoints
 * @param threadsUsed the actual number of threads that shall be used
 * @param workPerThread the amount of work per thread, this is calculated before the launch
 * in order to avoid integer division in the kernel and therefore save registers
 * @param D the main divisor that is used for the chunking
 */
void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results, int threadsUsed,
		int workPerThread, int D) {
	//for debug purposes ...
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);

	findBreakPointsFreeMode<<<numBlocks, blocksSize>>>(deviceRabin, deviceData, dataLen, results, threadsUsed, workPerThread, D);

	gpuErrchk(cudaGetLastError());

}

/**
 *
 * This function invokes the segmented chunking algorithm via a kernel launch. This launch
 * shall create breakpoints that fall within the minimum and maximum threshold and shall
 * also produces hashes of the content of every chunk that has been defined .
 *
 * @param blocksSize the number of threads within a block
 * @param numBlocks the number of blocks in the grid launch
 * @param rabinData_d pointer to the rabin data such as sliding window buffer and so on
 * @param ctx_d pointer to the chunking context holding information about the setting
 * @param dataToChunk_d pointer to the data to be chunked
 * @param sizeOfData size of the data to be chunked
 * @param activeThreads the number of actual active threads that will be performing the operation
 * @param hashed_d pointer to the memory in which the hashes shall be stored
 * @param results_d pointer to the array holding the resulting breakpoints
 */
void startSegmentedChunkingAndHashingKernel(size_t blocksSize, size_t numBlocks, rabinData* rabinData_d, chunkingContext* ctx_d, BYTE* dataToChunk_d,
		size_t sizeOfData, size_t activeThreads, BYTE* hashed_d, int* results_d) {

	//for debug purposes ...
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);

	findBreakPointsSegmented<<<numBlocks, blocksSize>>>(rabinData_d, ctx_d, dataToChunk_d, sizeOfData, results_d, activeThreads, hashed_d);

	gpuErrchk(cudaGetLastError());

}

#endif /* CHUNKINGKERNEL_CU_ */
