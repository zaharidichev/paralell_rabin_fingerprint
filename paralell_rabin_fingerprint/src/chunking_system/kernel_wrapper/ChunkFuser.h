/**
 * ChunkFuser.h
 *
 *	This class provides the functionality that is needed to fuse chunk breakpoints
 *	into final chunks. The process of chunk fusion is getting all the raw breakpoints
 *	found in a data stream and producing chunks that conform to minimal and maximum
 *	size. Additionally this class hashes all of those chunks (on the CPU). The functionality
 *	is used by the GPU chunker when continuous chunking is performed. Due to the non para-
 *	lell nature of the algorithm this type of fusion is done on the CPU instead on the GPU.
 *
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef CHUNKFUSER_H_
#define CHUNKFUSER_H_

#include "boost/smart_ptr.hpp"
#include <vector>
#include "Chunk.h"
#include "../GPU_code/BitFieldArray.h"
#include "../../etc/DedupDefines.h"
#include "string.h"
#include "../GPU_code/hashing/sha1_declarations.h"
#include <openssl/sha.h>
#include "../../etc/helpers/Macros.h"

class ChunkFuser {

private:
	size_t min; // the minimum size of a chunk that can be produced
	size_t max; // the maximum size of a chunk that can be produced
	size_t lastRealBP; // the last real breakpoint that was detected in a stream of chunks
	std::vector<boost::shared_ptr<Chunk> > fusedChunks; // the container that holds the fused chunks
	boost::shared_ptr<BYTE> residualData; // the residual data that needs to be accounted when the hash of the intermediate chunks is computed
	size_t residualSize; // the size of the residual data

	/**
	 * Given a bitfield array of raw points, this function produces a number of chunks
	 * that conform to the min - max threshold. It also accounts the position offset from
	 * the beginning of the file when it produces a chunk. This is needed since the raw
	 * bits only indicate where the breakpoints are in relation to the piece of data but
	 * not in relation to the whole file itself. If the file has been chunked in batches
	 * on the card, we need to add the position offset to the start and end positions of
	 *  every chunk.
	 *
	 * @param bitField the efficient bitfield array that holds the raw breakpoints
	 * @param sizeOfField the size of the bitfield array
	 * @param posOffset the position offset of this piece of data into the file
	 * @param hostBuffer the host buffer where the data is stored
	 */
	void fuseChunks(shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer);

	/**
	 * Given already created chunks, this function simply hashes their content in a serial
	 * fashion using a SHA1 algorithm.
	 *
	 * @param chunksToHash the container of chunks
	 * @param offset the offset into the file
	 * @param hostBuffer the buffer with host data
	 */
	void hashFusedChunks(std::vector<boost::shared_ptr<Chunk> > chunksToHash, size_t offset, BYTE* hostBuffer);

public:

	/**
	 * The main constructor for this class that takes as parameters the minimum and maximum size
	 * for a chunk
	 *
	 * @param min the minimum size of a chunk
	 * @param max the maximum size of a chunk
	 */
	ChunkFuser(size_t min, size_t max);

	/**
	 * Destructor ....
	 */
	virtual ~ChunkFuser();

	/**
	 * This function provides the main functionality for this class. Its key feature is that
	 * chunks can be added in portions the same way the are chunked on the GPU and the conti-
	 * nuity of their boundaries will be preserved even if a whole chunk goes over the boundary
	 * of a single batch onto another batch. This is done by keeping the last real breakpoint
	 * as a state in an object of this class and starting from it rather than the one that
	 * appears first in the raw breakpoints added. Along with this last "real" breakpoint,
	 * the data that spans from it to the end of the batch is kept so if a chunk that spans across
	 * the boundaries of two batches is found, this previous piece of data can be used to perform
	 * its hashing.
	 *
	 * @param bitField the raw breakpoints as a bit field array coming straight from the GPU
	 * @param sizeOfField the size of the bitfield array
	 * @param posOffset the position offset of the beginning of this batch relative to the whole file being chunked
	 * @param hostBuffer the host buffer that holds the data that was just chunked (used to extract the data for the production of hashes)
	 */
	void addRawBreakPoints(boost::shared_ptr<u_int32_t> bitField, size_t sizeOfField, size_t posOffset, BYTE* hostBuffer);

	/**
	 * Retrives the fused chunks.
	 * @return a collection of shared pointers to fused chunks
	 */
	std::vector<boost::shared_ptr<Chunk> > getChunks();

};

#endif /* CHUNKFUSER_H_ */
