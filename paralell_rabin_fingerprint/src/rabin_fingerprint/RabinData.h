/**
 * RabinData.h
 *
 * This file contains definitions of some useful data structures,
 * that are crucial to the implementation of the chunking algorithm
 *
 *
 *  Created on: Jun 8, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef RABINDATA_H_
#define RABINDATA_H_

//typedef uint64_t POLY_64;
//typedef Polynomial_128 POLY_128;


/**
 * This struct contains the push and pop pre - computed tables for the rabin
 * fingerprint algorithm. Along with that it, also contains the
 * irreducible polynomial that is used to calculate the final fingerprint
 */
typedef struct {
	POLY_64 Irreducble_PT;
	POLY_64 popTable[256]; // mod lookupTable
	POLY_64 pushTable[256]; // push lookupTable
	int shift; // size of shift when adding a byte
} rabinData;


/**
 * This struct is used to indicate the bounds of each thread in a cuda
 * kernel with respect to the data that it is working on.
 */
typedef struct threadBounds {
	int start;
	int end;
} threadBounds;


/**
 * This struct contains some of the main setting that are used in the
 * GPU chunking system such as main divisor, back divisor min and max
 * thresholds
 */
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
