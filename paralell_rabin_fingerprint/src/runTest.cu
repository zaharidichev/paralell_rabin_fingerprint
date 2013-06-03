/**
 * runTest.cpp
 *
 *
 *Just a little test to see whether we return the right values from the sliding FP code
 *This code launches a single kernel on the device and compute hashes for 0 - 255 bytes
 *
 *  Created on: Jun 3, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#include "stdint.h"
#include "stdio.h"
#include "kernelLauncher.cuh"
#define  SIZE 256

typedef uint64_t POLY_64;
typedef unsigned char BYTE;


void lineBreak()  {
	printf("\n");
}

void printResults(POLY_64* results) {
	for (int var = 0; var < SIZE; ++var) {
		printPolyAsHEXString(results[var]);
		lineBreak();
	}
}

void initHashData(BYTE* data) {
	for (int var = 0; var < SIZE; ++var) {
		data[var] = (BYTE)var;
	}
}


int main() {


	// value taken from LBFS source
	POLY_64 irreducible_PT = 0xbfe6b8a5bf378d83;
	BYTE dataToHash[SIZE];
	initHashData(dataToHash); // just write some data into the array
	POLY_64 results[SIZE];

	// this will fingerprint on the device and write results into the results array
	fingerprint256Bytes(results,dataToHash,irreducible_PT);


	printResults(results);

	return 0;
}
