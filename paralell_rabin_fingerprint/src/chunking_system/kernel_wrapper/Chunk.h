/**
 * Chunk.h
 *
 *	Represents a logical chunk of data that spans from a starting position to an ending
 *	position (not included) Additionally each chunk is associated with a hash values that
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

#define HEX( x ) std::setw(2) << std::setfill('0') << std::hex << (int)( x ) // simple macro for printing HEX to ostream
class Chunk {

private:
	size_t start; // the start position of the chunk
	size_t end; // the end position of the chunk
	size_t size; // the size
	boost::shared_ptr<BYTE> hash; //a pointer to the hash value of the chunk

public:
	/**
	 * The constructor to be used with this class, start and end positions need to be specified.
	 *
	 * @param start the starting position in a byte stream
	 * @param end the ending position in a byte stream
	 */
	Chunk(size_t start, size_t end) {
		this->start = start;
		this->end = end;
		this->size = this->end - this->start;
		this->hash = boost::shared_ptr<BYTE>((BYTE*) malloc(20));
	}

	/**
	 * Just our default destructor
	 */
	virtual ~Chunk() {

	}
	/**
	 * Getter for the start of the chunk
	 * @return the start
	 */
	size_t getStart() {
		return this->start;
	}
	/**
	 * Getter for the end of the chunk
	 * @return the end
	 */
	size_t getEnd() {
		return this->end;
	}
	/**
	 * Getter for the actual size of this chunk
	 * @return the size of the chunk
	 */
	size_t getSize() {
		return this->end - this->start;
	}
	/**
	 * Sets the hash of the chunk
	 * @param hash a pointer to an array of BYTE
	 */
	void setHash(boost::shared_ptr<BYTE> hash) {
		this->hash = hash;
	}
	/**
	 * Retrieves the hash of the chunk
	 * @return a pointer to an array of BYTE
	 */
	boost::shared_ptr<BYTE> getHash() {

		return this->hash;
	}

	/*
	 * We need this friend here in order to be able to pump it to
	 * an ostream :)
	 */
	friend std::ostream& operator <<(std::ostream& output, const Chunk& ch) {
		output << "[" << ch.start << " - " << ch.end << "] [" << ch.size << "] [";
		std::cout.setf(std::ios::hex, std::ios::basefield);

		for (int var = 0; var < 20; ++var) {
			std::cout << HEX(ch.hash.get()[var]);

		}

		std::cout.unsetf(std::ios::hex);

		output << "]";

		return output;
	}

};

#endif /* CHUNK_H_ */
