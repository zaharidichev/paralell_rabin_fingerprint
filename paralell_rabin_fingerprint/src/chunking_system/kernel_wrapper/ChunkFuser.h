/**
 * ChunkFuser.h
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */


#include "boost/smart_ptr.hpp"
#include <vector>
#include "Chunk.h"
#include "../GPU_code/BitFieldArray.h"
using namespace boost;
using namespace std;
#ifndef CHUNKFUSER_H_
#define CHUNKFUSER_H_

class ChunkFuser {
public:
	ChunkFuser(size_t min, size_t max);
	virtual ~ChunkFuser();
	void addRawBreakPoints(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset);
	vector<shared_ptr<Chunk> > getChunks();


private:
	size_t min;
	size_t max;
	size_t lastRealBP;
	vector<shared_ptr<Chunk> > fusedChunks;
	void fuseChunks(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset);

};

#endif /* CHUNKFUSER_H_ */
