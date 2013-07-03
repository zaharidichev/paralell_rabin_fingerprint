/**
 * GPUChunker.h
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef GPUCHUNKER_H_
#define GPUCHUNKER_H_
#include "../GPU_code/KernelStarter.h"
#include "../../etc/DedupDefines.h"
#include "ChunkContainer.h"
#include "string.h"
class GPUChunker {

private:
	POLY_64 irrPoly;
	int rabinDivisor;
	rabinData* rabinData_d;
	chunkCOntainer fuseChunks(bool* rawChunks, int min, int max, int dataLn);
	int getSizeOfBitArray(int dataLn);
public:
	GPUChunker(int RabinDivisor, POLY_64 irrPoly);
	virtual ~GPUChunker();
	chunkCOntainer chunkData(BYTE* dataToChunk, int dataLn, int minSize, int maxSize);
};

#endif /* GPUCHUNKER_H_ */
