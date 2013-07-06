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
//#include "/usr/local/cuda-5.0/samples/0_Simple/simplePrintf/cuPrintf.h"

__device__ inline void addBreakPoint(int* breakpoints, int pos, int *positionInArray) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < 2) {
		printf("%d,%d,%d\n", id, pos, (*positionInArray));

	}

	breakpoints[(*positionInArray)] = pos;
	(*positionInArray)++;
}

__device__ int getID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
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

__device__ void placeBreakPointBitsInField(bitFieldArray field, u_int32_t breakpoints, int pos) {
	field[pos] = breakpoints;
}

// flipping bool values:)
__device__ void flipSwitch(bool*swth) {
	*swth = !(*swth);
}

__device__ inline void chunkDataFreeMode(rabinData* deviceRabin, BYTE* data, threadBounds bounds, int D, bitFieldArray results, int activeThreads) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	//int pos = bounds.start; // the current position starting from a specific point
	//int lastBp = bounds.start; // the last breakpoint that was found

	//int backUpBp = 0; // the backup break point found by the secondary divisor

	//int positionInArray = bounds.BPwritePosition;

	//bool freeModeToggle = true;
	//bool initialChunkFoundToggle = true;

	//initting fingerprint...

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

			/*			if (getID() == activeThreads-1) {
			 printf("%d\n", pos);
			 }*/

			//results[pos] = true;
			//setReverseBit(u_int32_t* word, int x)
			setReverseBit(&partialBreakPoints, pos % 32);
		}

		if ((pos + 1) % 32 == 0 && pos != 0) {
			placeBreakPointBitsInField(results, partialBreakPoints, pos / 32);
			partialBreakPoints = 0;
		}
	}

	placeBreakPointBitsInField(results, partialBreakPoints, (bounds.end - 1) / 32);

	//printf("%d, %d, %d\n", getID(), bounds.start, bounds.end);

	//addBreakPoint(results, -1, &positionInArray);

// third phase
	/*
	 for (; pos < bounds.end; ++pos) {
	 fingerprint = update(deviceRabin, data[pos], fingerprint, &b);

	 if (pos - lastBp < ctx.minThr) {
	 continue;
	 }

	 if (bitMod(fingerprint, ctx.D) == ctx.D - 1) {

	 addBreakPoint(results, pos, &positionInArray);
	 lastBp = pos;
	 continue;
	 }

	 if (pos - lastBp < ctx.maxThr) {
	 continue;
	 } else {

	 addBreakPoint(results, pos, &positionInArray);

	 lastBp = pos;
	 }

	 }
	 */

/// bimodal chunking depricated for now ;0
	/*	for (; pos < bounds.end; ++pos) {
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
	 }*/

	//addBreakPoint(results, pos, &positionInArray);
//printResults_device(results,bounds);
}

#endif /* CHUNKER_H_ */

