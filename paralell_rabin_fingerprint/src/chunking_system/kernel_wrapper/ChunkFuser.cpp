/**
 * ChunkFuser.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ChunkFuser.h"

ChunkFuser::ChunkFuser(size_t min, size_t max) {
	this->residualSize = 0;
	this->min = min;
	this->max = max;
	this->lastRealBP = 0;
	this->fusedChunks = vector<shared_ptr<Chunk> >();
}

ChunkFuser::~ChunkFuser() {
	// TODO Auto-generated destructor stub
}

void ChunkFuser::addRawBreakPoints(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer, BYTE* dev_buffer) {

	fuseChunksExperimental(bitField, sizeOfField, posOffset, hostBuffer, dev_buffer);
}

void ChunkFuser::fuseChunks(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer) {

	int lastBP = this->lastRealBP;

	for (int i = 0; i < sizeOfField; ++i) {
		if (i - lastBP < min) {
			continue;
		}
		if (min <= i - lastBP <= max) {
			if (getBit(i, bitField.get())) {

				fusedChunks.push_back(shared_ptr<Chunk>(new Chunk(lastBP + posOffset, i + posOffset)));
				lastBP = i;
				continue;
			}

		}
		if (i - lastBP == max) {
			fusedChunks.push_back(shared_ptr<Chunk>(new Chunk(lastBP + posOffset, i + posOffset)));
			lastBP = i;
			continue;

		}

	}

	size_t residualDataSize = sizeOfField - lastBP;
	this->residualData = (BYTE*) malloc(sizeof(BYTE) * residualDataSize);

	memcpy(this->residualData, hostBuffer, residualDataSize);

	this->lastRealBP = lastBP - sizeOfField;
}

vector<shared_ptr<Chunk> > ChunkFuser::getChunks() {

	return this->fusedChunks;
}

void ChunkFuser::fuseChunksExperimental(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer, BYTE* devBuffer) {

	vector<shared_ptr<Chunk> > chunksForThisBatch;

	bool firstChunkMerged = false;
	int lastBP = this->lastRealBP;

	for (int i = 0; i < sizeOfField; ++i) {
		if (i - lastBP < min) {
			continue;
		}
		if (min <= i - lastBP <= max) {
			if (getBit(i, bitField.get())) {

				if (!firstChunkMerged && posOffset != 0) {
					shared_ptr<BYTE> scratchPad((BYTE*) malloc(this->residualSize + i));
					//what the hell am I doing.....there is no way this does not go wrong at some point ....

					std::cout << this->residualSize << " " << i << std::endl;
					memcpy(scratchPad.get(), this->residualData, this->residualSize);
					memcpy(scratchPad.get() + residualSize, hostBuffer, i);

					BYTE* digest = (BYTE*)malloc(20 * sizeof(BYTE));



					SHA1(scratchPad.get(), i + this->residualSize, digest);



					shared_ptr<Chunk> resolutionChunk(new Chunk(lastBP + posOffset, i + posOffset));
					resolutionChunk.get()->setHash(digest);
					this->fusedChunks.push_back(resolutionChunk);
					std::cout << "Hashed outside: " << *(resolutionChunk.get()) << std::endl;

					firstChunkMerged = true;
					lastBP = i;
					continue;
				}
				chunksForThisBatch.push_back(shared_ptr<Chunk>(new Chunk(lastBP + posOffset, i + posOffset)));
				//fusedChunks.push_back(shared_ptr<Chunk>(new Chunk(lastBP + posOffset, i + posOffset)));
				lastBP = i;
				continue;
			}

		}
		if (i - lastBP == max) {
			chunksForThisBatch.push_back(shared_ptr<Chunk>(new Chunk(lastBP + posOffset, i + posOffset)));
			lastBP = i;
			continue;

		}

	}

	this->hashFusedChunks(chunksForThisBatch, posOffset, hostBuffer,devBuffer);
	this->residualSize = sizeOfField - lastBP;
	this->residualData = (BYTE*) malloc(sizeof(BYTE) * residualSize);
	memcpy(this->residualData, hostBuffer + lastBP,this->residualSize);

	this->lastRealBP = lastBP - sizeOfField;

}

void ChunkFuser::hashFusedChunks(vector<shared_ptr<Chunk> > chunksToHash, size_t offset, BYTE* hostBuffer, BYTE* devBuffer) {
	std::cout << "hashing " << chunksToHash.size() << std::endl;

	 int numBreakpoints = chunksToHash.size() + 1;

	 size_t* breakpoints_h = (size_t*) malloc(sizeof(size_t) * numBreakpoints);


	 int cnt = 0;

	 for (std::vector<shared_ptr<Chunk> >::iterator it = chunksToHash.begin(); it != chunksToHash.end(); ++it) {
	 size_t begin = (*it).get()->getStart() ;
	 breakpoints_h[cnt] = begin - offset;
	 cnt++;
	 }



	 breakpoints_h[cnt + 1] = (chunksToHash.back().get()->getEnd()) - offset;




	 size_t* breakpoints_dev;
	 BYTE* hashes_dev;

	 CUDA_CHECK_RETURN(cudaMalloc(&breakpoints_dev, sizeof(size_t) * numBreakpoints));

	 CUDA_CHECK_RETURN(cudaMemcpy(breakpoints_dev, breakpoints_h, numBreakpoints * sizeof(size_t), cudaMemcpyHostToDevice));


	 CUDA_CHECK_RETURN(cudaMalloc(&hashes_dev,  20 * chunksToHash.size()));




	 createHashes(devBuffer, breakpoints_dev, hashes_dev, chunksToHash.size());


	 printf("Breakpoint\n");

	 BYTE* hashes = (BYTE*) malloc(20 * chunksToHash.size());
	 CUDA_CHECK_RETURN(cudaMemcpy(hashes, hashes_dev, 20 * chunksToHash.size(), cudaMemcpyDeviceToHost));

	 int cntH = 0;
	 for (std::vector<shared_ptr<Chunk> >::iterator it = chunksToHash.begin(); it != chunksToHash.end(); ++it) {
	 BYTE* hash = (BYTE*) malloc(20);

	 memcpy(hash, hashes + (cntH * 20), 20);

	 (*it).get()->setHash(hash);
	 cntH++;
	 }
	 CUDA_CHECK_RETURN(cudaFree(hashes_dev));
	 CUDA_CHECK_RETURN(cudaFree(breakpoints_dev));


/*	/////////
	for (std::vector<shared_ptr<Chunk> >::iterator it = chunksToHash.begin(); it != chunksToHash.end(); ++it) {
		size_t size = (*it).get()->getSize();
		size_t start = (*it).get()->getStart() - offset;
//		shared_ptr<BYTE> digest(new BYTE[20]);
		BYTE* hash = (BYTE*) malloc(20);
		SHA1(hostBuffer + (start), size, hash);
		(*it).get()->setHash(hash);

	}*/

	this->fusedChunks.insert(this->fusedChunks.end(), chunksToHash.begin(), chunksToHash.end());

}
