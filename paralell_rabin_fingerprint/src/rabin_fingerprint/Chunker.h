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
//#include "/usr/local/cuda-5.0/samples/0_Simple/simplePrintf/cuPrintf.h"

__device__ inline void addBreakPoint(int* breakpoints, int pos, int *positionInArray) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	printf("%d,%d,%d\n", id, pos, (*positionInArray));

	breakpoints[(*positionInArray)] = pos;
	(*positionInArray)++;
}

__device__ void printChunkData(int start, int end) {
	int id = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
	printf("[%d] Chunk defined:[%d --- %d] [%d]\n", start, end, end - start, id);
}

__device__ void printResults_device(int* results, threadBounds b) {
	for (int var = b.start; var < b.end; ++var) {
		printChunkData(results[var], results[var]);

	}

}

__device__ inline void chunkData(rabinData* deviceRabin, BYTE* data, threadBounds bounds, chunkingContext ctx, int* results) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	int pos = bounds.start; // the current position starting from a specific point
	int lastBp = bounds.start; // the last breakpoint that was found
	int backUpBp = 0; // the backup break point found by the secondary divisor

	int positionInArray = bounds.BPwritePosition;

	int avgSize = 0;
	int totalFound = 0;

	for (; pos < bounds.end; ++pos) {
		fingerprint = update(deviceRabin, data[pos], fingerprint, &b);

		if (pos - lastBp < ctx.minThr) {
			continue;
		}

		if (bitMod(fingerprint, ctx.Ddash) == ctx.Ddash - 1) {
			backUpBp = pos;
		}

		if (bitMod(fingerprint, ctx.D) == ctx.D - 1) {

			addBreakPoint(results, pos, &positionInArray);
			avgSize = avgSize + (pos - lastBp);
			totalFound++;
			backUpBp = 0;
			lastBp = pos;
			continue;
		}

		if (pos - lastBp < ctx.maxThr) {
			continue;
		}

		if (backUpBp != 0) {

			addBreakPoint(results, backUpBp, &positionInArray);

			avgSize = avgSize + (backUpBp - lastBp);

			totalFound++;
			lastBp = backUpBp;
			backUpBp = 0;
		} else {

			addBreakPoint(results, pos, &positionInArray);

			avgSize = avgSize + (pos - lastBp);

			totalFound++;
			lastBp = pos;
			backUpBp = 0;
		}
	}

	addBreakPoint(results, pos, &positionInArray);
	//printResults_device(results,bounds);
}

#endif /* CHUNKER_H_ */

