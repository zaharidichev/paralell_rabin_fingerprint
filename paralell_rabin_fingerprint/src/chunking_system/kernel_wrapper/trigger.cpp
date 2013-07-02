/**
 * trigger.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "GPUChunker.h"

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
	int sizeOfData = 536870912;
	int minSize = 32768;
	int maxSize = 131072;

	BYTE* dataToChunk = allocateData(sizeOfData);

	GPUChunker chunker = GPUChunker(512, 0xbfe6b8a5bf378d83);

	chunkCOntainer chunks = chunker.chunkData(dataToChunk, sizeOfData, minSize, maxSize);

	printf("REAL CHUNKS\n");
	for (int var = 0; var < chunks.size; ++var) {
		printf("%d\n", chunks.breakpoints[var]);
	}

	free(dataToChunk);
	free(chunks.breakpoints);

}