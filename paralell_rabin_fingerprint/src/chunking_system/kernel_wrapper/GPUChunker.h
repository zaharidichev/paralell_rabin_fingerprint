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
#include "../GPU_code/BitFieldArray.h"
#include "boost/shared_ptr.hpp"
#include "Chunk.h"
#include <vector>
#include "ChunkFuser.h"
#include "../GPU_code/ResourceManagement.h"
#include <iostream>     // std::cin, std::cout
#include <fstream>      // std::ifstream
using namespace std;
using namespace boost;

class GPUChunker {

private:
	POLY_64 irrPoly;
	int rabinDivisor;
	size_t minSize;
	size_t maxSize;
	rabinData* rabinData_d;
	chunkCOntainer fuseChunks(bitFieldArray rawChunks, int min, int max, int dataLn);
	int getSizeOfBitArray(int dataLn);
	shared_ptr<ChunkFuser> fuser;
public:
	GPUChunker(int RabinDivisor, POLY_64 irrPoly, size_t minSize, size_t maxSize);
	virtual ~GPUChunker();
	vector<shared_ptr<Chunk> > chunkData(BYTE* dataToChunk, size_t dataLn);
	vector<shared_ptr<Chunk> > chunkDataFromFile(ifstream& file, size_t dataLn);

	vector<shared_ptr<Chunk> > chunkAndHashDataFromFile(ifstream& file, size_t dataLn);

};

#endif /* GPUCHUNKER_H_ */
