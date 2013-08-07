/**
 * Chunker.h
 *
 *
 * This file contains the implementation of the two main chunking function -
 * the one for continuous chunking and the one for segmented one. Along with that
 * helper functions are provided for further abstraction. ALl these functions
 * are declared as __device__ ones since they all are executed from within kernels
 * that run on the GPU hardware.
 *
 *  Created on: Jun 7, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef CHUNKER_H_
#define CHUNKER_H_

#include <stdio.h>
#include <stdlib.h>
#include "../rabin_fingerprint/RabinFingerprint.h"
#include "cuda_runtime.h"
#include "../chunking_system/GPU_code/BitFieldArray.h"
#include "../chunking_system/GPU_code/hashing/sha1_kernel.cu"

/**
 * Retrieves the global ID of the current thread.
 *
 * @return the id of the thread
 */
__device__ int getID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 *
 * This function is used to add a breakpoint in the integer array that holds the breakpoints
 * for the segmented chunking approach. In addition to that the function also increments the
 * pointer into the array by one.
 *
 * @param breakpoints the pointer to the breakpoints array
 * @param pos the value of the breakpoint
 * @param positionInArray  the position into the array (pointer)
 */
__device__ inline void addBreakPointSimple(int* breakpoints, int pos, int *positionInArray) {

	breakpoints[(*positionInArray)] = pos;
	(*positionInArray)++; // incrmenet pointer
}

/**
 *
 * This function is used to place a 32 bit figure that represents 32 breakpoints
 * into the bitfield array used by the continuous chunking approach to store raw
 * breakpoints positions
 *
 * @param field  the bit field array
 * @param breakpoints the 32 breakpoints as a 32 bit integer
 * @param pos the position into which that figure needs to go
 */
__device__ void addBreakPointsInBitArray(bitFieldArray field, u_int32_t breakpoints, int pos) {
	field[pos] = breakpoints;
}

/**
 *
 * This function performs segmented chunking and computes hashes for every created chunk
 * All of the results are stored in buffers on the global memory of the device. The particular
 * algorithm that is used for this segmented chunking is the Two - threshold , two - divisors
 * one, which is proposed by HP:
 *
 * http://www.hpl.hp.com/techreports/2005/HPL-2005-30R1.pdf
 *
 * @param deviceRabin pointer to the device rabin data (push and pop tables,etc.)
 * @param data the data that is to be fingerprinted
 * @param bounds the bounds of the particular thread (from where to where it needs to fingerprint)
 * @param ctx the chunking context holding the settings for the system
 * @param results the resulting breakpoints
 * @param activeThreads the number of active threads
 * @param hashes the buffer that should hold the hashes
 */
__device__ inline void chunkDataWithLimits(rabinData* deviceRabin, BYTE* data, threadBounds bounds, chunkingContext* ctx, int* results, int activeThreads,
		BYTE* hashes) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used
	int skipper = 0;
	int pos = bounds.start + skipper; // the current position starting from a specific point
	int lastBp = bounds.start + skipper; // the last breakpoint that was found
	int backUpBp = 0; // the backup break point found by the secondary divisor

	int positionInBPArray = ctx->BpreakpointsPerThread * getID();

	if (getID() != 0) {
		positionInBPArray++;
	}


	if (getID() == 0) {

		addBreakPointSimple(results, pos, &positionInBPArray);
	}
	//Calculate the starting position in the hash array for this thread
	int positionInHashArray = ctx->BpreakpointsPerThread * getID() * 20;

	for (; pos < bounds.end; ++pos) {
		//iterate through all the data

		fingerprint = update(deviceRabin, data[pos], fingerprint, &b); // push another byte into the fingerprint

		if (pos - lastBp < ctx->minThr) {
			// if we are below the min threshold just go on to the next iteration

			continue;
		}

		if (bitMod(fingerprint, ctx->Ddash) == ctx->Ddash - 1) {
			/*
			 * if a breakpoint is found by the backup divisor,
			 * then save this one for now...
			 */

			backUpBp = pos;

		}

		if (bitMod(fingerprint, ctx->D) == ctx->D - 1) {
			/*
			 * if a breakpoint is found by the main divisor,
			 * then add this one
			 */

			addBreakPointSimple(results, pos, &positionInBPArray);
			//also compute the hash and palce it into the array
			sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);

			positionInHashArray = positionInHashArray + 20; // forward the hash position by 20 (the size of sha1 hash in bytes)

			// continous to next iteration
			backUpBp = 0;
			lastBp = pos;
			pos = pos + skipper;
			continue;
		}

		if (pos - lastBp < ctx->maxThr) {
			// if nothing is found but we are below max thrshold just continoue
			continue;
		}

		if (backUpBp != 0) {

			/*
			 * at this point if we are above the max and
			 * no breakpoint was found by the main or the
			 * backup divisor just impsoe a hard threshold
			 *
			 */
			addBreakPointSimple(results, backUpBp, &positionInBPArray);
			sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);
			positionInHashArray = positionInHashArray + 20;

			lastBp = backUpBp;
			backUpBp = 0;
			pos = pos + skipper;

		} else {
			/*
			 * but if there is a backup breakpoint,
			 * use it to define one and compute hash for the resulting chunk
			 */

			addBreakPointSimple(results, pos, &positionInBPArray);
			sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);
			positionInHashArray = positionInHashArray + 20;
			lastBp = pos;
			backUpBp = 0;
			pos = pos + skipper;

		}
	}

	//at the end we need to add the last breakpoint
	addBreakPointSimple(results, pos, &positionInBPArray);
	sha1_internal(data + lastBp, pos - lastBp, hashes + positionInHashArray);

}

/**
 *
 * THis function performs free - mode fingeprinting, recording all found breakpoints
 * no matter whether they are within minimum or maximum threshold. All the rest is left
 * to the  CPU. For this reason the efficient bitfield structure is used to store the
 * breakpoints as bits in 32bit wide figure ints.
 *
 * @param deviceRabin the device rabin data such as push and pop tables
 * @param data the data to be fingerprinted
 * @param bounds the bounds for the particular thread
 * @param D the divisor for the fingerprinting
 * @param results a pointer to the bitfield array holding the results
 * @param activeThreads the number of active threads
 */
__device__ inline void chunkDataFreeMode(rabinData* deviceRabin, BYTE* data, threadBounds bounds, int D, bitFieldArray results, int activeThreads) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0; // the fingerprint that will be used

	if (getID() != 0) {

		for (int var = bounds.start - 48; var < bounds.start; ++var) {
			fingerprint = update(deviceRabin, data[var], fingerprint, &b);

		}

	}

	//first phase

	u_int32_t partialBreakPoints = 0; // we hold this in registers for 32 breakpoints before writing into the array
	for (int pos = bounds.start; pos < bounds.end; ++pos) { // for all data
		fingerprint = update(deviceRabin, data[pos], fingerprint, &b); // update the  fingeprint

		if (bitMod(fingerprint, D) == D - 1) {
			// if it matches the condition, the set the bit into the 32 bit figure

			setReverseBit(&partialBreakPoints, pos % 32);
		}

		if ((pos + 1) % 32 == 0 && pos != 0) {
			/* if at this iteration the current 32 bit figure has been used up,
			 * write it to the bitfield array in global memory
			 */

			addBreakPointsInBitArray(results, partialBreakPoints, pos / 32);
			partialBreakPoints = 0;
		}
	}
	//add the last breakpoint
	addBreakPointsInBitArray(results, partialBreakPoints, (bounds.end - 1) / 32);

}

#endif /* CHUNKER_H_ */

