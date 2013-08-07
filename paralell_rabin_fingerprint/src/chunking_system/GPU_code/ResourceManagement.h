/**
 * ResourceManagement.h
 *
 *
 * This is a collection of functions that wrap CUDA driver API calls for managing resources.
 * on the device. Functionality includes allocating entities of data that are specific to the
 * application domain of chunking on the GPU.
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include <stdio.h>
#include "../../etc/helpers/Macros.h"
#include "cuda_runtime.h"
#include "../../rabin_fingerprint/RabinFingerprint.h"
#ifndef FUNKYFUNKS_H_
#define FUNKYFUNKS_H_

#define MIN_WORK_PER_THREAD 262144

/**
 *
 * This function initializes the chunking context on the device and returns a pointer
 * to device memory which can be passed to a kernel invocation
 *
 * @param minThr the minimum threshold
 * @param maxThr the maximum threshold
 * @param breakpointsPerThread the number of breakpoints per thread (words case)
 * @param sizeOfBreakpointsArray the size of the breapoints array
 *
 * @return device memory pointer to the chunking context
 */
inline chunkingContext* initChunkingContextOnDevice(size_t minThr, size_t maxThr, size_t breakpointsPerThread, size_t sizeOfBreakpointsArray) {

	chunkingContext ctx;
	ctx.workPerThread = MIN_WORK_PER_THREAD;
	ctx.D = 512;
	ctx.Ddash = 256;
	ctx.minThr = minThr;
	ctx.maxThr = maxThr;
	ctx.BpreakpointsPerThread = breakpointsPerThread;
	ctx.sizeOfBreakpointsArray = sizeOfBreakpointsArray;

	chunkingContext* ctx_d;

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &ctx_d, sizeof(chunkingContext)));
	//copy the data to device
	CUDA_CHECK_RETURN(cudaMemcpy(ctx_d, &ctx, sizeof(chunkingContext), cudaMemcpyHostToDevice));
	return ctx_d;
}

/**
 *
 * Given a pointer to memory containing SHA-1 hashed and a pointer to
 * host memory to which these hashes need to be copied, this function
 * makes the necessary CUDA calls to transfer the data and handle potential
 * errors
 *
 * @param hashes_d pointer to device memory where hashes reside
 * @param hashes_h pointer to host memory where hashes need to be copied to
 * @param numHashes the number of hashes to be copied
 * @param sizeOfHashes the size of a single hash in bytes
 */
inline void downloadHashesFromDeviceToHost(BYTE* hashes_d, BYTE* hashes_h, size_t numHashes, size_t sizeOfHashes) {
	CUDA_CHECK_RETURN(cudaMemcpy(hashes_h, hashes_d, sizeof(BYTE) * numHashes * sizeOfHashes, cudaMemcpyDeviceToHost));

}

/**
 *
 * This function just abstracts away the flushing of the hash buffer on the device side.
 *
 * @param hashes_d pointer to device memory
 * @param numHashes number of hashes
 * @param sizeOfHashes the size of a single SHA-1 hash
 */
inline void flushHashesBuffer(BYTE* hashes_d, size_t numHashes, size_t sizeOfHashes) {
	CUDA_CHECK_RETURN(cudaMemset(hashes_d, 0, sizeof(BYTE) * numHashes * sizeOfHashes));

}

/**
 * Function to abstract away the allocation process of a device side buffer for SHA-1 hashes
 *
 * @param numHashes the number of hashes needed
 * @param sizeOfHashes the size of a single hash
 * @return the pointer to the allocated memory
 */
inline BYTE* allocateHashesBufferOnDevice(size_t numHashes, size_t sizeOfHashes) {
	BYTE* hashes_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &hashes_d, sizeof(BYTE) * numHashes * sizeOfHashes));
	CUDA_CHECK_RETURN(cudaMemset(hashes_d, 0, sizeof(BYTE) * numHashes * sizeOfHashes));
	return hashes_d;
}

/**
 * Flushes the breakpoint buffer on the device. This is needed when new kernel
 * launch is dispatched
 *
 * @param breakpoints_d pointer to device memory
 * @param size size of the breakpoints buffer (number of integers )
 */
inline void flushBreakpointsBufferOnDevice(int* breakpoints_d, size_t size) {
	CUDA_CHECK_RETURN(cudaMemset(breakpoints_d, 0, sizeof(int) * size));

}

/**
 *
 * Allocates a breakpoints buffer on the device.
 * @param size size of the buffer (number of potential breakpoints )
 *
 * @return pointer to device memory
 */
inline int* allocateBreakpointsBufferOnDevice(size_t size) {
	int* resultingBreakpoints_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &resultingBreakpoints_d, sizeof(int) * size));
	CUDA_CHECK_RETURN(cudaMemset(resultingBreakpoints_d, 0, sizeof(int) * size));
	return resultingBreakpoints_d;
}

/**
 * Given a device   and host pointers, the breakpoint buffer is downlaoded
 * from the device to the host memory
 *
 * @param breakpointsBuffer_d pointer to device memory
 * @param breakpointsBuffer_h pointer to host memory
 *
 * @param size size of the buffer
 */
inline void downloadBreakpointsBufferFromDevice(int* breakpointsBuffer_d, int* breakpointsBuffer_h, size_t size) {
	CUDA_CHECK_RETURN(cudaMemcpy(breakpointsBuffer_h, breakpointsBuffer_d, sizeof(int) * size, cudaMemcpyDeviceToHost));

}

/**
 * Calculates the number of breakpoints needed for a specific size of
 * an array of data that will be fingerprinted
 *
 * @param dataLn the length of the data to be fingerprinted
 * @param minThreshold the minimum threshold of a chunks size
 * @return the maximum number of breakpoints that can be found in the data.
 */
inline int getSizeOFBreakpointsArray(int dataLn, int minThreshold) {
	return (dataLn % minThreshold == 0) ? (dataLn / minThreshold) + 1 : (dataLn / minThreshold) + 2;

}

/**
 * Allocates the fingepritnting window and push/pop tables on the device.
 * @param irreduciblePolynomial a irreducible polynomial in the form of a 64 bit figure
 * @return a pointer to the data allocated on the device
 */
inline rabinData* allocateDeviceRabinData(POLY_64 irreduciblePolynomial) {
	rabinData hostData;
	rabinData* deviceData;
	initWindow(&hostData, irreduciblePolynomial);
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &deviceData, sizeof(rabinData)));
	//copy the data to device
	CUDA_CHECK_RETURN(cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));

	return deviceData;

}

/**
 * Returns the determined minimum work per thread in terms of data length in bytes
 * @return the minimum work per thread
 */
inline size_t getMinWorkPerThread() {
	return MIN_WORK_PER_THREAD;
}

/**
 * This method is a wrapper around the cuda driver API. It provides functionality for allocating a buffer of a
 * specified size on the device
 * @param size the size of the data that needs to be allocated
 * @return a pointer to the allocated data .
 */
inline BYTE* allocateDeviceBuffer(size_t size) {
	BYTE* bufferPtr_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &bufferPtr_d, sizeof(BYTE) * size))
	return bufferPtr_d;
}

/**
 * Given a pointers to already allocated memory on the host and the device , the function transfers the data from the host
 * to the device, adhering to the specified offsets in the data.
 *
 * @param hostData pointer to the data allocated on the host
 * @param devBuffer pointer to the data residing on the device
 * @param offset offset into the data on the host
 * @param size the size of data that needs to be copied.
 */
inline void uploadDataToDeviceBuffer(BYTE* hostData, BYTE* devBuffer, size_t offset, size_t size) {
	hostData = hostData + offset;
	CUDA_CHECK_RETURN(cudaMemcpy(devBuffer, hostData, sizeof(BYTE) * size, cudaMemcpyHostToDevice));
}

/**
 * Given an irreducibly polynomial PT, the function initializes the Rabin device data onto the GPU.
 *
 * @param irrPoly an irreducible polynomial in the form of a 64 bit integer
 * @return pointer to the data on the GPU
 */
inline rabinData* initRabinDataOnDevice(POLY_64 irrPoly) {
	rabinData hostData;
	rabinData* deviceData;
	initWindow(&hostData, irrPoly);
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &deviceData, sizeof(rabinData)));
	CUDA_CHECK_RETURN(cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));
	return deviceData;

}
inline int getBlockSize() {
	return 160;
}
inline int getNumBlocks(int threadsNeeded) {
	int blockSize = getBlockSize();
	int numBlocks = threadsNeeded / blockSize;
	if (threadsNeeded % blockSize) {
		++numBlocks;
	}
	return numBlocks;
}

/**
 *This function determines the number of threads needed for a particular data - parallel job to be
 *run on the device. Given an amount of work per thread.
 *
 * @param dataLn the length of the data in bytes
 * @param minWorkPerThread the minimum length of the data that each thread is allocated
 * @return the number of threads needed to perform the job
 */
inline int getNumNeededThreads(int dataLn, int minWorkPerThread) {
	//int minWorkPerThread = 262144;

	//return (dataLn % minSizePerThread != 0) ? (dataLn / minSizePerThread + 1) : dataLn / minSizePerThread;
	return (dataLn <= getMinWorkPerThread()) ? 1 : dataLn / minWorkPerThread;
}

/**
 * A templated function which purpose is to free a memory resource that was allocated on the GPU. It can take
 * a pointer of any type.
 *
 * @param resrcPtr the pointer to the memory allocated on the device
 */
template<typename T> void freeCudaResource(T* resrcPtr) {
	CUDA_CHECK_RETURN(cudaFree(resrcPtr));
}

/**
 * The function uses a simple calculation to determine the device buffer that can be allocated. The granulation
 * of allocation is 64 MB. The function tries to allocate as much memory as possible, leaving a minimum of 64 MB
 *
 * @return the size of memory that could be allocated on the device buffer.
 */
inline size_t getDeviceBufferSize() {

	size_t granulationSize = 67108864;
	int device;
	cudaGetDevice(&device);
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	size_t sizeOfBuffer = ((properties.totalGlobalMem / granulationSize) - 1) * granulationSize;
	return sizeOfBuffer;
}

#endif /* FUNKYFUNKS_H_ */
