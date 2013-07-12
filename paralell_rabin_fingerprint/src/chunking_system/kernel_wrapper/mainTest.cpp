/**
 * mainTest.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "GPUChunker.h"
#include "../GPU_code/BitFieldArray.h"
typedef unsigned char BYTE;

bool compareChunks(std::vector<shared_ptr<Chunk> > &left, std::vector<shared_ptr<Chunk> > &right) {

	int size = left.size();

	for (int var = 0; var < size; ++var) {
		shared_ptr<Chunk> ch1 = left[var];
		shared_ptr<Chunk> ch2 = right[var];

		if (!(ch1.get()->getStart() == ch2.get()->getStart()) || !(ch1.get()->getEnd() == ch2.get()->getEnd())) {
			return false;
		}

		assert(ch1.get()->getEnd() == ch2.get()->getEnd());

		for (int var_2 = 0; var_2 < 20; ++var_2) {
			if (!(ch1.get()->getHash().get()[var_2] == ch2.get()->getHash().get()[var_2])) {
				return false;
			}
		}

	}
	return true;
}

int main() {

	size_t sizeOfDataToFingerprint = 262144;
	size_t minSize = 32768;
	size_t maxSize = 131072;
	int primaryDiv = 512;
	int secondaryDiv = 256;
	POLY_64 irr_Poly = 0xbfe6b8a5bf378d83;

	std::ifstream infile("../../../resources/testData.dat", std::ofstream::binary);

	GPUChunker chunker = GPUChunker(primaryDiv, secondaryDiv, 0xbfe6b8a5bf378d83, minSize, maxSize);

	std::vector<shared_ptr<Chunk> > chunks_continuous = chunker.chunkFileFromDisk(infile, sizeOfDataToFingerprint, CONTINUOUS);
	infile.seekg(0, infile.beg); // roll back the file stream
	std::vector<shared_ptr<Chunk> > chunks_segmeted = chunker.chunkFileFromDisk(infile, sizeOfDataToFingerprint, SEGMENTED);

	printf((compareChunks(chunks_continuous, chunks_segmeted)) ? "Conformance test passed...\n" : "Conformance test failed...\n");
}
