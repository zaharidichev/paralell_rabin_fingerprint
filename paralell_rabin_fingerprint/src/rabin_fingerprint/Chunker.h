/**
 * Fingerprinter.cu
 *
 *  Created on: Jun 7, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef CHUNKER_H_
#define CHUNKER_H_

#include <stdio.h>
#include <stdlib.h>
#include "../rabin_fingerprint/RabinFingerprint.h"
#include "cuda_runtime.h"
#include "../chunking_system/GPU_code/BitFieldArray.h"
#include "../chunking_system/GPU_code/hashing/sha1_kernel.cu"
//#include "/usr/local/cuda-5.0/samples/0_Simple/simplePrintf/cuPrintf.h"

__device__ int getID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline void addBreakPointSimple(int* breakpoints, int pos, int *positionInArray) {

	breakpoints[(*positionInArray)] = pos;
	(*positionInArray)++;
}

__device__ void addBreakPointsInBitArray(bitFieldArray field, u_int32_t breakpoints, int pos) {
	field[pos] = breakpoints;
}

__device__ inline void chunkDataWithLimits(rabinData* deviceRabin, BYTE* data, threadBounds bounds, chunkingContext* ctx, int* results, int activeThreads,
		BYTE* hashes) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	int pos = bounds.start; // the current position starting from a specific point
	int lastBp = bounds.start; // the last breakpoint that was found
	int backUpBp = 0; // the backup break point found by the secondary divisor

	int positionInBPArray = ctx->BpreakpointsPerThread * getID();

	if (getID() != 0) {
		positionInBPArray++;
	}

	if (getID() == 0) {
		addBreakPointSimple(results, pos, &positionInBPArray);
	}
	int positionInHashArray = ctx->BpreakpointsPerThread * getID() * 20;

	for (; pos < bounds.end; ++pos) {

		fingerprint = update(deviceRabin, data[pos], fingerprint, &b);

		if (pos - lastBp < ctx->minThr) {

			continue;
		}

		if (bitMod(fingerprint, ctx->Ddash) == ctx->Ddash - 1) {

			backUpBp = pos;

		}

		if (bitMod(fingerprint, ctx->D) == ctx->D - 1) {

			addBreakPointSimple(results, pos, &positionInBPArray);

			sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);

			positionInHashArray = positionInHashArray + 20;

			backUpBp = 0;
			lastBp = pos;
			continue;
		}

		if (pos - lastBp < ctx->maxThr) {
			continue;
		}

		if (backUpBp != 0) {

			addBreakPointSimple(results, backUpBp, &positionInBPArray);
			sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);
			positionInHashArray = positionInHashArray + 20;

			lastBp = backUpBp;
			backUpBp = 0;
		} else {

			addBreakPointSimple(results, pos, &positionInBPArray);
			sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);
			positionInHashArray = positionInHashArray + 20;
			lastBp = pos;
			backUpBp = 0;
		}
	}

	addBreakPointSimple(results, pos, &positionInBPArray);
	sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);

}

__device__ inline void chunkDataFreeMode(rabinData* deviceRabin, BYTE* data, threadBounds bounds, int D, bitFieldArray results, int activeThreads) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	if (getID() != 0) {

		for (int var = bounds.start - 48; var < bounds.start; ++var) {
			fingerprint = update(deviceRabin, data[var], fingerprint, &b);

		}

	}

	//first phase

	u_int32_t partialBreakPoints = 0;
	for (int pos = bounds.start; pos < bounds.end; ++pos) {
		fingerprint = update(deviceRabin, data[pos], fingerprint, &b);

		if (bitMod(fingerprint, D) == D - 1) {

			setReverseBit(&partialBreakPoints, pos % 32);
		}

		if ((pos + 1) % 32 == 0 && pos != 0) {
			addBreakPointsInBitArray(results, partialBreakPoints, pos / 32);
			partialBreakPoints = 0;
		}
	}

	addBreakPointsInBitArray(results, partialBreakPoints, (bounds.end - 1) / 32);

}

#endif /* CHUNKER_H_ */

