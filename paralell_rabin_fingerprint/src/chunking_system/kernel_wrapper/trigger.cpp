/**
 * trigger.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "GPUChunker.h"
#include "../GPU_code/BitFieldArray.h"

typedef unsigned char BYTE;
BYTE* allocateData(int size) {

	BYTE* data = (BYTE*) malloc(sizeof(BYTE) * size);
	srand(2);
	for (int var = 0; var < size; ++var) {
		data[var] = (BYTE) rand() % 256;
	}

	return data;

}

int main() {

	int sizeOfData = 737233947;
	int minSize = 32768;
	int maxSize = 131072;

	//const char* dataToChunk = (const char*)allocateData(sizeOfData);

	std::ifstream infile("/home/zahari/Desktop/2.6_kernels_merged.dat", std::ofstream::binary);

	GPUChunker chunker = GPUChunker(512, 0xbfe6b8a5bf378d83, minSize, maxSize);
	std::vector<shared_ptr<Chunk> > chunks = chunker.chunkFile_segmented(infile, sizeOfData, minSize, maxSize);

	for (std::vector<shared_ptr<Chunk> >::iterator it = chunks.begin(); it != chunks.end(); ++it) {
		std::cout << *((*it).get()) << std::endl;
	}

	printf("----------------------------\n");

	unsigned char* buffer = (unsigned char*) malloc(15859);

	infile.seekg(737218088);

	infile.read((char*) buffer, 15859);
	BYTE* digest = (BYTE*) malloc(20);

	SHA1(buffer, 15859, digest);

	for (int var = 0; var < 20; ++var) {
		printf("%02x", digest[var]);

	}
	printf("\n");

}
