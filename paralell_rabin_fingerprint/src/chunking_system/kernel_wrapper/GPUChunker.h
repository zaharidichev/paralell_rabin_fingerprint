/**
 * GPUChunker.h
 *
 *	This class serves as the main wrapper that provides functionality to the GPU
 *	accelerated chunking system. The client of this class need not know any of the
 *	Implementation details. All it takes is parameters regarding the minimum and
 *	maximum chunk size as well as the irreducible constant used by the fingerpritning
 *	algorithm. The class is designed to work with a  file stream and can handle files
 *	of arbitrary size without being restricted by the global memory available on the GPU.
 *
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef GPUCHUNKER_H_
#define GPUCHUNKER_H_
#include "../GPU_code/KernelStarter.h"
#include "../../etc/DedupDefines.h"
#include "string.h"
#include "../GPU_code/BitFieldArray.h"
#include "boost/shared_ptr.hpp"
#include "Chunk.h"
#include <vector>
#include "ChunkFuser.h"
#include "../GPU_code/ResourceManagement.h"
#include "../IO_tools/FileReader.h"
#include <iostream>
#include <fstream>
#include <cmath>

/**
 * This is simply a type, which purpose is to set the type of chunking that will be performed
 */
enum ChunkingMethod {
	SEGMENTED, CONTINUOUS
};

class GPUChunker {

private:
	POLY_64 irrPoly; // the irreducible polynomial that will be used for fingerprinting
	int rabinDivisorPrime; // the main divisor of the fingerprinting algorithm
	int rabinDivisorSecond; // the secondary divisor
	size_t minSize; // the minimum size of the chunks that will be produced
	size_t maxSize; // the maximum size of the chunks that will be produces
	rabinData* rabinData_d; // a pointer to the rabin fingerprinting context that is initialized on the device
	shared_ptr<ChunkFuser> fuser;

	/**
	 * This function is used internally to number of batches that are needed in order to
	 * fingerprint the whole file. Fingerprinting is done in batches that are as big as the
	 * buffer that is allocated on the device .
	 *
	 * @param dataLn the length of the data
	 * @param GPUbufferSize the size of the buffer that is available on the device
	 * @return the number of batches that will be needed to fingerprint this data
	 */
	size_t getNumberOfBatches(size_t dataLn, size_t GPUbufferSize);

	/**
	 *	This function is used in order to determine the current length of the batch that is
	 *	being worked on. The function is useful when it comes to working on sizes that are
	 *	not devisible by the size of the buffer. In those cases the last bath is the residual
	 *	of this division and needs to be accounted for.
	 *
	 * @param iteration the current iteration of the batch dispatch loop
	 * @param bufferSize the size of the buffer in bytes
	 * @param numBatches the number of batches that are needed to finish the data
	 * @param dataLn the length of the data
	 * @return the size of the current batch
	 */
	size_t getLengthOfCurrentBatch(int itteration, size_t bufferSize, size_t numBatches, size_t dataLn);

	/**
	 *
	 * This private method is used for chunking and fingeprinting in a segmented fashion. What that means
	 * is that the size of chunks will be affected by how much data is dedicated per thread on the GPU.
	 * This avoids the chunk fusion process and accelerates the performance of the algorithm greatly. The
	 * tradeoff is that by changing the amount of work per thread , the size way chunks are formed will be
	 * changed thus producing different chunks every time. Experiments show that on highly versioned data
	 * and relatively small chunk sizes (4k - 16k) efficiency of deduplication is deteriorated only by roughly 5%
	 *
	 * @param file the filestream
	 * @param dataLn the length of the file
	 * @return
	 */
	std::vector<boost::shared_ptr<Chunk> > chunkFile_segmented(FileReader& file, size_t dataLn);

	/**
	 * This private method performs what is called continuous chunking. It is a two - phase process in which
	 * the free - mode chunking (without restricting min and max size) is performed on the GPU. This way
	 * all the possible breakpoints are recorded in an efficient bitfield array on the GPU. After that the
	 * data is copied and the chunks are fused on the CPU (and hashed) in order to conform to the min-
	 * imum and maximum chunk sizes specified in the constructor of this class. Although, slower, this method
	 * guarantees that chunks will always have unique boundaries, no matter what is the granularity of work
	 * per thread.
	 *
	 * @param file
	 * @param dataLn
	 * @return
	 */
	std::vector<boost::shared_ptr<Chunk> > chunkFile_continuous(FileReader& file, size_t dataLn);

public:
	/** Constructor that initializes the chunker
	 *
	 * @param rabinDivisor the primary rabin divisor used to determine breakpoints
	 * @param rabinDivisorSecond the secondary divisor that finds breakpoints with higher probability
	 * @param irrPoly the irreducible constant that is used to perform fingerprinting
	 * @param minSize the minimum size of a chunk
	 * @param maxSize the maximum size of a chunk
	 */
	GPUChunker(int rabinDivisor, int rabinDivisorSecond, POLY_64 irrPoly, size_t minSize, size_t maxSize);

	/**
	 * Destructor... Nothing to be freed, since only shared pointers are used...
	 */
	virtual ~GPUChunker();

	/**
	 * This is the method that is public and provides a way to use either the segmented
	 * or the continuous chunking by specifying a parameter that indicated the particular
	 * method.
	 *
	 * @param file a filestream
	 * @param dataLn the length of the file
	 * @param method the chunking method that needs to be used
	 * @return a collection of pointers to Chunk objects
	 */
	std::vector<boost::shared_ptr<Chunk> > chunkFileFromDisk(FileReader& file, size_t dataLn, ChunkingMethod enumChunkingMethod);

};

#endif /* GPUCHUNKER_H_ */
