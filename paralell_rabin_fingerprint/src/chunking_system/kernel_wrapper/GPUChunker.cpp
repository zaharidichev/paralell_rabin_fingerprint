/**
 * GPUChunker.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "GPUChunker.h"
#include "../GPU_code/ResourceManagement.h"

GPUChunker::GPUChunker(int RabinDivisor, POLY_64 irrPoly) {
	this->irrPoly = irrPoly;
	this->rabinDivisor = rabinDivisor;
	this->rabinData_d = initRabinDataOnDevice(irrPoly);
}

GPUChunker::~GPUChunker() {
}

chunkCOntainer GPUChunker::fuseChunks(bool* rawChunks, int min, int max, int dataLn) {
	int maxSizeOfBPArrayNeeded = (dataLn / min) + 1;

	int* fusedPoints = (int*) malloc(sizeof(int) * maxSizeOfBPArrayNeeded);
	fusedPoints[0] = 0;
	int posInFusedArray = 1;
	int lastBP = 0;

	for (int i = 0; i < dataLn; ++i) {
		if (i - lastBP < min) {
			continue;
		}
		if (min <= i - lastBP <= max) {
			if (rawChunks[i]) {
				fusedPoints[posInFusedArray] = i;
				posInFusedArray++;
				lastBP = i;
				continue;
			}

		}
		if (i - lastBP == max) {
			fusedPoints[posInFusedArray] = i;
			posInFusedArray++;
			lastBP = i;
			continue;

		}

	}

	fusedPoints[posInFusedArray] = dataLn;
	posInFusedArray++;
	int* truncatedArray = (int*) malloc(sizeof(int) * (posInFusedArray));

	memcpy(truncatedArray, fusedPoints, sizeof(int) * posInFusedArray);

	free(fusedPoints);

	chunkCOntainer result;
	result.size = posInFusedArray;
	result.breakpoints = truncatedArray;

	return result;
}

chunkCOntainer GPUChunker::chunkData(BYTE* dataToChunk, int dataLn, int minSize, int maxSize) {
	int minWork = 262144;

	// we first need to upload the data to the device :)
	BYTE* dataToFingerprint_d = uploadDataToDevice(dataToChunk, dataLn);

	//allocate space for the raw breakpoints;

	bool* rawBreakPoints_d = allocateBPArrayOnDevice(dataLn);

	// calcualte thread heuristics
	int threadsNeeded = getNumNeededThreads(dataLn, minWork);

	int blocksize = 160;

	int numBlocks = threadsNeeded / blocksize;
	if (threadsNeeded % blocksize) {
		++numBlocks;
	}
	//start kernel
	startCreateBreakpointsKernel(blocksize, numBlocks, this->rabinData_d, dataToFingerprint_d, dataLn, rawBreakPoints_d, threadsNeeded, minWork, 512);

	//lets downlaod our raw results now...
	bool* rawBreakPoints_h = downloadBreakPointFromDevice(rawBreakPoints_d, dataLn);

	//and free the resources allocated
	freeCudaResource(rawBreakPoints_d);
	freeCudaResource(dataToFingerprint_d);

	return fuseChunks(rawBreakPoints_h, minSize, maxSize, dataLn);
}
