/**
 * KernelStarter.h
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef KERNELSTARTER_H_
#define KERNELSTARTER_H_
#include "../../etc/DedupDefines.h"
#include "../../rabin_fingerprint/RabinData.h"
#include "BitFieldArray.h"
extern "C" void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results,
		int threadsUsed, int workPerThread, int D);

extern "C" void startSegmentedChunkingAndHashingKernel(size_t blocksSize, size_t numBlocks, rabinData* rabinData_d, chunkingContext* ctx_d, BYTE* dataToChunk_d,
		size_t sizeOfData, size_t activeThreads, BYTE* hashed_d, int* results_d);

#endif /* KERNELSTARTER_H_ */
