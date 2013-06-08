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



  __device__ inline  void chunkData(rabinData* deviceRabin, BYTE* data, long from, long to, int D, int Ddash, int minChSize, int maxChSize) {



	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	long pos = from; // the current position starting from a specific point
	long lastBp = 0; // the last breakpoint that was found
	long backUpBp = 0; // the backup break point found by the secondary divisor



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
			cuPrintf("Found breakpoint at: %lu [%lu] [%d]\n", pos, fingerprint,
					pos - lastBp);
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

			cuPrintf("Found breakpoint at: %lu [%lu] [%d]\n", pos, fingerprint,
					backUpBp - lastBp);
			avgSize = avgSize + (backUpBp - lastBp);

			totalFound++;
			lastBp = backUpBp;
			backUpBp = 0;
		} else {
			cuPrintf("Found breakpoint at: %lu [%lu] [%d]\n", pos, fingerprint,
					pos - lastBp);
			avgSize = avgSize + (pos - lastBp);

			totalFound++;
			lastBp = pos;
			backUpBp = 0;
		}
	}

	cuPrintf("Found on device: %d , Avg size: %d ", totalFound,
			avgSize / totalFound);
}

#endif /* CHUNKER_H_ */


