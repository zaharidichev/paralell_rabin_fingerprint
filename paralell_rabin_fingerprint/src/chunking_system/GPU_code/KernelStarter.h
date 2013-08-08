/**
 * KernelStarter.h
 *
 * This file provides two functions that are abstractions
 * over the raw CUDA kernel launches. Those functions are used to start
 * the two type of chunking on the GPU device - the continuous type and
 * the segmented type chunking.
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSTARTER_H_
#define KERNELSTARTER_H_
#include "../../etc/DedupDefines.h"
#include "../../rabin_fingerprint/RabinData.h"
#include "BitFieldArray.h"

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
extern "C" void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results,
		int threadsUsed, int workPerThread, int D);

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
extern "C" void startSegmentedChunkingAndHashingKernel(size_t blocksSize, size_t numBlocks, rabinData* rabinData_d, chunkingContext* ctx_d, BYTE* dataToChunk_d,
		size_t sizeOfData, size_t activeThreads, BYTE* hashed_d, int* results_d);

#endif /* KERNELSTARTER_H_ */
