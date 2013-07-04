/**
 * Chunk.cpp
 *
 *  Created on: Jul 4, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "Chunk.h"



Chunk::Chunk(size_t start, size_t end) {
	this->start = start;
	this->end = end;
	this->size = this->end - this->start;
}

Chunk::~Chunk() {

}
#include <iostream>
using namespace std;
size_t Chunk::getStart() {
	return this->start;
}

size_t Chunk::getEnd() {
	return this->end;
}

size_t Chunk::getSize() {
	return this->end - this->start;
}

ostream& operator << (ostream& output, const Chunk& ch) {
	output << "[" << ch.start << " - " << ch.end << "] [" << ch.size << "]";
	return output;
}
