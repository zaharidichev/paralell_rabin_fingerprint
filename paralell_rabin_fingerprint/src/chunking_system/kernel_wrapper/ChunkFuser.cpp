/**
 * ChunkFuser.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "ChunkFuser.h"

ChunkFuser::ChunkFuser(size_t min, size_t max) {

	this->min = min;
	this->max = max;
	this->lastRealBP = 0;
	this->fusedChunks = vector<shared_ptr<Chunk> >();
}

ChunkFuser::~ChunkFuser() {
	// TODO Auto-generated destructor stub
}

void ChunkFuser::addRawBreakPoints(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset) {

	fuseChunks(bitField, sizeOfField, posOffset);
}

void ChunkFuser::fuseChunks(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset) {

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

	this->lastRealBP = lastBP - sizeOfField;
}

vector<shared_ptr<Chunk> > ChunkFuser::getChunks() {
	return this->fusedChunks;
}
