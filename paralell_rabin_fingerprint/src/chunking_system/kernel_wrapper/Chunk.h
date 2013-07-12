/**
 * Chunk.h
 *
 *	Represents a logical chunk of data that spans from a starting position to an ending
 *	position (not included) Additionally each chunk is associated with a hash valeus that
 *	identifies it
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef CHUNK_H_
#define CHUNK_H_
#include <iostream>
#include <boost/shared_ptr.hpp>
#include "../../etc/DedupDefines.h"
#include <iomanip>
#include "stdio.h"

#define HEX( x ) setw(2) << setfill('0') << hex << (int)( x ) // simple macro for printing HEX to ostream
class Chunk {

private:
	size_t start; // the start position of the chunk
	size_t end; // the end position of the chunk
	size_t size; // the size
	boost::shared_ptr<BYTE> hash; //a pointer to the hash value of the chunk

public:
	/**
	 * The constructor to be used with this class, start and end positions need to be specified.
	 * @param start the starting position in a byte stream
	 * @param end the ending position in a byte stream
	 */
	Chunk(size_t start, size_t end);
	/**
	 * Just our default destructor
	 */
	virtual ~Chunk();
	/**
	 * Getter for the start of the chunk
	 * @return the start
	 */
	size_t getStart();
	/**
	 * Getter for the end of the chunk
	 * @return the end
	 */
	size_t getEnd();
	/**
	 * Getter for the actual size of this chunk
	 * @return the size of the chunk
	 */
	size_t getSize();
	/**
	 * Sets the hash of the chunk
	 * @param hash a pointer to an array of BYTE
	 */
	void setHash(boost::shared_ptr<BYTE> hash);
	/**
	 * Retrieves the hash of the chunk
	 * @return a pointer to an array of BYTE
	 */
	boost::shared_ptr<BYTE> getHash();
	friend std::ostream& operator <<(std::ostream& output, const Chunk& ch);

};

#endif /* CHUNK_H_ */
