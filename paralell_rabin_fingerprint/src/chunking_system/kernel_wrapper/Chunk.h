/**
 * Chunk.h
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include <iostream>
using namespace std;
#ifndef CHUNK_H_
#define CHUNK_H_

class Chunk {

private:
	size_t start;
	size_t end;
	size_t size;
public:
	Chunk(size_t start, size_t end);
	virtual ~Chunk();
	size_t getStart();
	size_t getEnd();
	size_t getSize();
	friend ostream &operator<<(ostream &output, const Chunk &ch);

};

#endif /* CHUNK_H_ */
