/**
 * Chunk.h
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include <iostream>
#include <boost/shared_ptr.hpp>
#include "../../etc/DedupDefines.h"
#include <iomanip>
#include "stdio.h"


using namespace std;
using namespace boost;
#ifndef CHUNK_H_
#define CHUNK_H_
#define HEX( x ) setw(2) << setfill('0') << hex << (int)( x )



class Chunk {

private:
	size_t start;
	size_t end;
	size_t size;
	BYTE* hash;
public:
	Chunk(size_t start, size_t end);
	virtual ~Chunk();
	size_t getStart();
	size_t getEnd();
	size_t getSize();
	void setHash(BYTE* hash);
	BYTE* getHash();
	friend ostream& operator <<(ostream& output, const Chunk& ch);

};

#endif /* CHUNK_H_ */
