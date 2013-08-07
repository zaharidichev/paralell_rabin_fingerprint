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
	this->fusedChunks = std::vector<boost::shared_ptr<Chunk> >();
}

ChunkFuser::~ChunkFuser() {
	// TODO Auto-generated destructor stub
}

void ChunkFuser::addRawBreakPoints(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer) {
	fuseChunks(bitField, sizeOfField, posOffset, hostBuffer);
}

std::vector<boost::shared_ptr<Chunk> > ChunkFuser::getChunks() {

	return this->fusedChunks;
}

void ChunkFuser::fuseChunks(boost::shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer) {

	std::vector<boost::shared_ptr<Chunk> > chunksForThisBatch;

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

					//std::cout << this->residualSize << " " << i << std::endl;
					memcpy(scratchPad.get(), this->residualData.get(), this->residualSize);
					memcpy(scratchPad.get() + residualSize, hostBuffer, i);

					boost::shared_ptr<BYTE> digest((BYTE*) malloc(20 * sizeof(BYTE)));
					SHA1(scratchPad.get(), i + this->residualSize, digest.get());

					shared_ptr<Chunk> resolutionChunk(new Chunk(lastBP + posOffset, i + posOffset));
					resolutionChunk.get()->setHash(digest);
					this->fusedChunks.push_back(resolutionChunk);
					//std::cout << "Hashed outside: " << *(resolutionChunk.get()) << std::endl;

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

	this->hashFusedChunks(chunksForThisBatch, posOffset, hostBuffer);
	this->residualSize = sizeOfField - lastBP;
	this->residualData = shared_ptr<BYTE>((BYTE*) malloc(sizeof(BYTE) * residualSize));
	memcpy(this->residualData.get(), hostBuffer + lastBP, this->residualSize);

	this->lastRealBP = lastBP - sizeOfField;

}

void ChunkFuser::hashFusedChunks(std::vector<boost::shared_ptr<Chunk> > chunksToHash, size_t offset, BYTE* hostBuffer) {

	for (std::vector<shared_ptr<Chunk> >::iterator it = chunksToHash.begin(); it != chunksToHash.end(); ++it) {
		size_t size = (*it).get()->getSize();
		size_t start = (*it).get()->getStart() - offset;
		shared_ptr<BYTE> digest(new BYTE[20]);
		SHA1(hostBuffer + (start), size, digest.get());
		(*it).get()->setHash(digest);

	}

	this->fusedChunks.insert(this->fusedChunks.end(), chunksToHash.begin(), chunksToHash.end());

}
