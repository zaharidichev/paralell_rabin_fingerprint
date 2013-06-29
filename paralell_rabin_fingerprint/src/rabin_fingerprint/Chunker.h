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
#include "/usr/local/cuda-5.0/samples/0_Simple/simplePrintf/cuPrintf.h"

__device__ inline void addBreakPoint(long* breakpoints, long pos,
		long *positionInArray) {
	breakpoints[(*positionInArray)] = pos;
	(*positionInArray)++;
}

__device__ void printChunkData(long start, long end) {
	cuPrintf("Chunk defined:[%d --- %d] [%d]\n", start, end, end - start);
}

__device__ void printResults_device(long* results) {

	long avg_size = 0;
	long cnt = 1;
	while (results[cnt] != -1) {
		printChunkData(results[cnt-1], results[cnt]);
		avg_size = avg_size + (results[cnt] - results[cnt-1]);
		cnt++;
	}
	cuPrintf("Chunks on GPU: %d , Avg size: %d\n", cnt,
			avg_size / cnt);
}


__device__ inline void chunkData(rabinData* deviceRabin, BYTE* data, long from,
		long to, int D, int Ddash, int minChSize, int maxChSize,
		long* results) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	long pos = from; // the current position starting from a specific point
	long lastBp = 0; // the last breakpoint that was found
	long backUpBp = 0; // the backup break point found by the secondary divisor

	long resultsCounter = 0;
	addBreakPoint(results, pos, &resultsCounter);

	long avgSize = 0;
	int totalFound = 0;

	for (; pos < to; ++pos) {
		fingerprint = update(deviceRabin, data[pos], fingerprint, &b);

		if (pos - lastBp < minChSize) {
			continue;
		}

		if (bitMod(fingerprint, Ddash) == Ddash - 1) {
			backUpBp = pos;
		}

		if (bitMod(fingerprint, D) == D - 1) {

			addBreakPoint(results, pos, &resultsCounter);
			avgSize = avgSize + (pos - lastBp);
			totalFound++;
			backUpBp = 0;
			lastBp = pos;
			continue;
		}

		if (pos - lastBp < maxChSize) {
			continue;
		}

		if (backUpBp != 0) {

			addBreakPoint(results, backUpBp, &resultsCounter);

			avgSize = avgSize + (backUpBp - lastBp);

			totalFound++;
			lastBp = backUpBp;
			backUpBp = 0;
		} else {

			addBreakPoint(results, pos, &resultsCounter);

			avgSize = avgSize + (pos - lastBp);

			totalFound++;
			lastBp = pos;
			backUpBp = 0;
		}
	}

	addBreakPoint(results, -1, &resultsCounter);
	printResults_device(results);
}

#endif /* CHUNKER_H_ */

