/**
 * GPUChunker.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "GPUChunker.h"

GPUChunker::GPUChunker(int RabinDivisor, POLY_64 irrPoly, size_t minSize, size_t maxSize) {
	this->irrPoly = irrPoly;
	this->rabinDivisor = rabinDivisor;
	this->rabinData_d = initRabinDataOnDevice(irrPoly);
	this->minSize = minSize;
	this->maxSize = maxSize;
	this->fuser = shared_ptr<ChunkFuser>(new ChunkFuser(this->minSize, this->maxSize));

}

GPUChunker::~GPUChunker() {
}

chunkCOntainer GPUChunker::fuseChunks(bitFieldArray rawChunks, int min, int max, int dataLn) {
	printf("HELLLOOOOO");

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
			if (getBit(i, rawChunks)) {
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

int GPUChunker::getSizeOfBitArray(int dataLn) {
	return (dataLn % 32 == 0) ? dataLn / 32 : (dataLn / 32) + 1;

}

vector<shared_ptr<Chunk> > GPUChunker::chunkData(BYTE* dataToChunk, size_t dataLn) {
	int minWorkPerThread = 262144;

	int binaryDataBuffer_d = min(getDeviceBufferSize(), dataLn);

	//min()
	// calcualte the number of kernels that need to be started to compelte the data
	int numBatches = dataLn / binaryDataBuffer_d;
	if (dataLn % binaryDataBuffer_d != 0) {
		numBatches++;
	}

	//allocate buffers for binary data and raw fingerprints...
	BYTE* dataToFingerprint_d = allocateDeviceBuffer(binaryDataBuffer_d);
	int numberOfBitWordsNeeded = getSizeOfBitArray(binaryDataBuffer_d);
	bitFieldArray raw_results_d = createBitFieldArrayOnDevice(numberOfBitWordsNeeded);

	for (int i = 0; i < numBatches; ++i) {
		// loop as many times as needed for all the data to be fingerpritned

		int lengthOfBatch = binaryDataBuffer_d; // calcualte the current lenght of data that will be worked on by the kernel

		if (i == numBatches - 1 && dataLn % binaryDataBuffer_d != 0) {
			lengthOfBatch = dataLn % binaryDataBuffer_d;
		}
		printf("Batch_size: %d\n", lengthOfBatch);

		// uplaod the data from the binary array to the device buffer
		uploadDataToDeviceBuffer(dataToChunk, dataToFingerprint_d, +(i * lengthOfBatch), lengthOfBatch);
		//flush the fingeprrint buffer..
		flushBitfieldFufferOnDevice(raw_results_d, numberOfBitWordsNeeded);

		//calculate launch parameters
		int threadsNeeded = getNumNeededThreads(lengthOfBatch, minWorkPerThread);
		int blocksize = 160;
		int numBlocks = threadsNeeded / blocksize;
		if (threadsNeeded % blocksize) {
			++numBlocks;
		}

		//start kernel
		startCreateBreakpointsKernel(blocksize, numBlocks, this->rabinData_d, dataToFingerprint_d, lengthOfBatch, raw_results_d, threadsNeeded,
				minWorkPerThread, 512);

		//calculate how many breakpoints need to be downladoed from the buffer, based on the lenght of the data fingerpinted by the kernel
		int numberOfBitWordsNeeded = getSizeOfBitArray(lengthOfBatch);

		//get the raw pointes and feed the mto the fuser
		shared_ptr<u_int32_t> results(downloadBitFieldArrayFromDevice(numberOfBitWordsNeeded, raw_results_d));
		(*fuser.get()).addRawBreakPoints(results, lengthOfBatch, i * lengthOfBatch);

	}

	//at the end free the buffers allocated on the device

	freeCudaResource(raw_results_d);
	freeCudaResource(dataToFingerprint_d);
	// and return the chunks produced by the fuser
	return this->fuser.get()->getChunks();

}

vector<shared_ptr<Chunk> > GPUChunker::chunkDataFromFile(ifstream& file, size_t dataLn) {

	int minWorkPerThread = 262144;
	int bufferSize = min(getDeviceBufferSize(), dataLn);

	int numBatches = dataLn / bufferSize;
	if (dataLn % bufferSize != 0) {
		numBatches++;
	}

	BYTE* hostBuffer = (BYTE*)malloc(sizeof(BYTE) * bufferSize);
	BYTE* deviceBuffer = allocateDeviceBuffer(bufferSize);
	int numberOfBitWordsNeeded = getSizeOfBitArray(bufferSize);
	bitFieldArray raw_results_d = createBitFieldArrayOnDevice(numberOfBitWordsNeeded);


	for (int i = 0; i < numBatches; ++i) {

		int lengthOfBatch = bufferSize; // calcualte the current lenght of data that will be worked on by the kernel
		if (i == numBatches - 1 && dataLn % bufferSize != 0) {
			lengthOfBatch = dataLn % bufferSize;
		}
		//read into buffer ....
		for (int var = 0; var < lengthOfBatch; ++var) {
			hostBuffer[var] = file.get();
		}


		// uplaod the data from the binary array to the device buffer
		uploadDataToDeviceBuffer(hostBuffer, deviceBuffer,0, lengthOfBatch);
		//flush the fingeprrint buffer..
		flushBitfieldFufferOnDevice(raw_results_d, numberOfBitWordsNeeded);

		//calculate launch parameters
		int threadsNeeded = getNumNeededThreads(lengthOfBatch, minWorkPerThread);
		int blocksize = 160;
		int numBlocks = threadsNeeded / blocksize;
		if (threadsNeeded % blocksize) {
			++numBlocks;
		}

		//start kernel
		startCreateBreakpointsKernel(blocksize, numBlocks, this->rabinData_d, deviceBuffer, lengthOfBatch, raw_results_d, threadsNeeded,
				minWorkPerThread, 512);

		//calculate how many breakpoints need to be downladoed from the buffer, based on the lenght of the data fingerpinted by the kernel
		int numberOfBitWordsNeeded = getSizeOfBitArray(lengthOfBatch);

		//get the raw pointes and feed the mto the fuser
		shared_ptr<u_int32_t> results(downloadBitFieldArrayFromDevice(numberOfBitWordsNeeded, raw_results_d));
		(*fuser.get()).addRawBreakPoints(results, lengthOfBatch, i * lengthOfBatch);

	}

	freeCudaResource(raw_results_d);
	freeCudaResource(deviceBuffer);
	// and return the chunks produced by the fuser
	return this->fuser.get()->getChunks();


}
