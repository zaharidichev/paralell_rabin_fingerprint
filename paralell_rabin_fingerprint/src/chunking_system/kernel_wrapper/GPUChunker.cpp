/**
 * GPUChunker.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "GPUChunker.h"

GPUChunker::GPUChunker(int rabinDivisor, int rabinDivisorSecond, POLY_64 irrPoly, size_t minSize, size_t maxSize) {
	this->irrPoly = irrPoly;
	this->rabinDivisorPrime = rabinDivisorPrime;
	this->rabinDivisorSecond = rabinDivisorSecond;
	this->rabinData_d = initRabinDataOnDevice(irrPoly);
	this->minSize = minSize;
	this->maxSize = maxSize;
	this->fuser = shared_ptr<ChunkFuser>(new ChunkFuser(this->minSize, this->maxSize));

}

GPUChunker::~GPUChunker() {
}

std::vector<boost::shared_ptr<Chunk> > GPUChunker::chunkFileFromDisk(std::ifstream& file, size_t dataLn, ChunkingMethod enumChunkingMethod) {

	return (enumChunkingMethod) ? this->chunkFile_continuous(file, dataLn) : this->chunkFile_segmented(file, dataLn);

}

std::vector<boost::shared_ptr<Chunk> > GPUChunker::chunkFile_continuous(std::ifstream& file, size_t dataLn) {

	int minWorkPerThread = getMinWorkPerThread();
	int bufferSize = std::min(getDeviceBufferSize(), dataLn);
	int numBatches = getNumberOfBatches(dataLn, bufferSize);

	boost::shared_ptr<BYTE> hostBuffer((BYTE*) malloc(sizeof(BYTE) * bufferSize));
	BYTE* deviceBuffer = allocateDeviceBuffer(bufferSize);
	int numberOfBitWordsNeeded = getSizeOfBitArray(bufferSize);
	bitFieldArray raw_results_d = createBitFieldArrayOnDevice(numberOfBitWordsNeeded);

	for (int i = 0; i < numBatches; ++i) {

		int lengthOfBatch = getLengthOfCurrentBatch(i, bufferSize, numBatches, dataLn);

		//read into buffer ....

		file.read((char*) hostBuffer.get(), lengthOfBatch);

		// uplaod the data from the binary array to the device buffer
		uploadDataToDeviceBuffer(hostBuffer.get(), deviceBuffer, 0, lengthOfBatch);
		//flush the fingeprrint buffer..
		flushBitfieldFufferOnDevice(raw_results_d, numberOfBitWordsNeeded);

		//calculate launch parameters

		int threadsNeeded = getNumNeededThreads(lengthOfBatch, minWorkPerThread);
		int blockSize = getBlockSize();
		int numBlocks = getNumBlocks(threadsNeeded);

		//start kernel
		startCreateBreakpointsKernel(blockSize, numBlocks, this->rabinData_d, deviceBuffer, lengthOfBatch, raw_results_d, threadsNeeded, minWorkPerThread, 512);

		//calculate how many breakpoints need to be downladoed from the buffer, based on the lenght of the data fingerpinted by the kernel
		int numberOfBitWordsNeeded = getSizeOfBitArray(lengthOfBatch);

		//get the raw pointes and feed the mto the fuser
		shared_ptr<u_int32_t> results(downloadBitFieldArrayFromDevice(numberOfBitWordsNeeded, raw_results_d));
		(*fuser.get()).addRawBreakPoints(results, lengthOfBatch, i * lengthOfBatch, hostBuffer.get());

	}

	freeCudaResource(raw_results_d);
	freeCudaResource(deviceBuffer);
	// and return the chunks produced by the fuser
	return this->fuser.get()->getChunks();

}

std::vector<boost::shared_ptr<Chunk> > GPUChunker::chunkFile_segmented(std::ifstream& file, size_t dataLn) {
	std::vector<boost::shared_ptr<Chunk> > finalChunks = std::vector<boost::shared_ptr<Chunk> >();

	rabinData* rabinDataOnDevice = initRabinDataOnDevice(this->irrPoly);

	size_t minWorkPerThread = getMinWorkPerThread();
	//size_t bufferSize = min(getDeviceBufferSize(), dataLn);
	size_t bufferSize = 134217728 / 2;

	size_t numBatches = getNumberOfBatches(dataLn, bufferSize);

	boost::shared_ptr<BYTE> hostBuffer((BYTE*) malloc(sizeof(BYTE) * bufferSize));
	BYTE* deviceBuffer = allocateDeviceBuffer(bufferSize);

	size_t sizeOfBreakPointsBuffer = getSizeOFBreakpointsArray(bufferSize, this->minSize);

	int* breakPoints_d = allocateBreakpointsBufferOnDevice(sizeOfBreakPointsBuffer);

	boost::shared_ptr<int> breakpoints_host((int*) malloc(sizeOfBreakPointsBuffer * sizeof(int)));

	boost::shared_ptr<BYTE> hashesBuffer_host((BYTE*) malloc(sizeOfBreakPointsBuffer * 20 * sizeof(BYTE)));

	BYTE* hashesBuffer_d = allocateHashesBufferOnDevice(sizeOfBreakPointsBuffer, 20);

	int bpsPerThread = round(((double) (sizeOfBreakPointsBuffer)) / getNumNeededThreads(bufferSize, minWorkPerThread));

	chunkingContext* ctx_d = initChunkingContextOnDevice(this->minSize, this->maxSize, bpsPerThread, sizeOfBreakPointsBuffer);

	for (int i = 0; i < numBatches; ++i) {

		int currentLengthOfBatch = getLengthOfCurrentBatch(i, bufferSize, numBatches, dataLn);
		//read into buffer
		file.read((char*) hostBuffer.get(), currentLengthOfBatch);

		uploadDataToDeviceBuffer(hostBuffer.get(), deviceBuffer, 0, currentLengthOfBatch);
		flushBreakpointsBufferOnDevice(breakPoints_d, sizeOfBreakPointsBuffer);
		flushHashesBuffer(hashesBuffer_d, sizeOfBreakPointsBuffer, 20);
		int threadsNeeded = getNumNeededThreads(currentLengthOfBatch, minWorkPerThread);
		int blockSize = getBlockSize();
		int numBlocks = getNumBlocks(threadsNeeded);
		//printf("%d %d\n", blockSize, numBlocks);

		startSegmentedChunkingAndHashingKernel(blockSize, numBlocks, rabinDataOnDevice, ctx_d, deviceBuffer, currentLengthOfBatch, threadsNeeded,
				hashesBuffer_d, breakPoints_d);

		downloadHashesFromDeviceToHost(hashesBuffer_d, hashesBuffer_host.get(), sizeOfBreakPointsBuffer, 20);
		downloadBreakpointsBufferFromDevice(breakPoints_d, breakpoints_host.get(), sizeOfBreakPointsBuffer);

		if (currentLengthOfBatch != bufferSize) {
			sizeOfBreakPointsBuffer = getSizeOFBreakpointsArray(currentLengthOfBatch, this->minSize);
		}

		size_t offsetFromStartOfFile = i * bufferSize;

		for (int var = 1; var < sizeOfBreakPointsBuffer; ++var) {

			shared_ptr<Chunk> chunk(new Chunk(breakpoints_host.get()[var - 1] + offsetFromStartOfFile, breakpoints_host.get()[var] + offsetFromStartOfFile));

			boost::shared_ptr<BYTE> hashForCurrentChunk((BYTE*) malloc(20));
			memcpy(hashForCurrentChunk.get(), hashesBuffer_host.get() + ((var - 1) * 20), 20);
			chunk.get()->setHash(hashForCurrentChunk);
			finalChunks.push_back(chunk);

		}

	}

	freeCudaResource(hashesBuffer_d);
	freeCudaResource(deviceBuffer);
	freeCudaResource(breakPoints_d);
	freeCudaResource(ctx_d);
	freeCudaResource(rabinDataOnDevice);

	return finalChunks;

}

size_t GPUChunker::getNumberOfBatches(size_t dataLn, size_t GPUbufferSize) {
	int numBatches = dataLn / GPUbufferSize;

	return (dataLn % GPUbufferSize != 0) ? ++numBatches : numBatches;

}

size_t GPUChunker::getLengthOfCurrentBatch(int itteration, size_t bufferSize, size_t numBatches, size_t dataLn) {

	size_t lengthOfBatch = bufferSize;
	if (itteration == numBatches - 1 && dataLn % bufferSize != 0) {
		lengthOfBatch = dataLn % bufferSize;
	}
	return lengthOfBatch;
}

