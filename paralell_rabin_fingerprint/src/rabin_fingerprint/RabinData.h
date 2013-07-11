/**
 * RabinData.h
 *
 *  Created on: Jun 8, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef RABINDATA_H_
#define RABINDATA_H_

//typedef uint64_t POLY_64;
//typedef Polynomial_128 POLY_128;

typedef struct {
	POLY_64 Irreducble_PT;
	POLY_64 popTable[256]; // mod lookupTable
	POLY_64 pushTable[256]; // push lookupTable
	int shift; // size of shift when adding a byte
} rabinData;


typedef struct threadBounds {
	int start;
	int end;
} threadBounds;

typedef struct chunkingContext {
	int D;
	int Ddash;
	int minThr;
	int maxThr;
	int workPerThread;
	int sizeOfBreakpointsArray;
	int BpreakpointsPerThread;

} chunkingContext;

#endif /* RABINDATA_H_ */
