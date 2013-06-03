/**
 * RabinFingerprint.cu
 *
 *Implementation of functions defined in RabinFingerprint.cuh
 *
 *  Created on: May 31, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef RABINFINGERPRINT_CUH_
#define RABINFINGERPRINT_CUH_
#include "../math/BitOps.cu"
#include "../math/PolyMath.cu"
#include "../data_structures/Buffer.cu"
#include "RabinFingerprint.cuh"


inline  __host__  void initWindow(rabinData* window, POLY_64 PT) {
	window->Irreducble_PT = PT; // set the internal variable to the irreducible poly
	precomputeTables(window);
}

inline __host__  void precomputeTables(rabinData* fingerprintData) {

	//fingerprintData->fingerprint = 0;
	int fDegree = degree(fingerprintData->Irreducble_PT);
	fingerprintData->shift = fDegree - 8;
	long T1 = mod((INT_64(1) << fDegree), fingerprintData->Irreducble_PT);
	for (INT_64 j = 0; j < 256; j++) {
		// computing the T table
		fingerprintData->pushTable[(int) j] = (polyModmult(j, T1,
				fingerprintData->Irreducble_PT) | (j << fDegree));
		//printPolyAsHEXString(fingerprintData->pushTable[(int) j]);
		//printf("\n");
	}
	//printf("\n");
	INT_64 sizeshift = 1;
	for (INT_64 i = 1; i < WIN_SIZE; i++)
		sizeshift = pushAByte(sizeshift, fingerprintData, (BYTE) 0);
	for (INT_64 i = 0; i < 256; i++) {
		fingerprintData->popTable[i] = polyModmult(i, sizeshift,
				fingerprintData->Irreducble_PT);
		//printPolyINHEX(fingerprintData->U[(int) i]);
		//printf(" ");
	}
}

inline __host__   __device__ POLY_64 popAByte(rabinData* data, BYTE byte,
		POLY_64 oldFingerprint, byteBuffer* buffer) {
	/*
	 * it might seem deceiving that we need a new byte in our POP function
	 * but the fact is that this is part of the mechanism for updating
	 * the fingerprint. we need to add a new byte to the window so we can
	 * slide it and get the byte that falls out of it
	 */

	// get the byte that falls of the window
	BYTE old = push(byte, buffer);
	/*
	 * remove its contribution from the fingerprint. Note that we do not add the
	 * contribution of the new byte to the fingerprint. This is done via the
	 * pushAByte( ) function. Those two functions are supposed to work together
	 */
	POLY_64 newFingerprint = oldFingerprint ^ data->popTable[old & 0xFF];

	return newFingerprint;
}

inline __host__     __device__ INT_64 pushAByte(INT_64 oldFingerprint,
		rabinData* data, BYTE byte) {

	// updates the fingerprint by adding contribution of the byte pushed
	int idexInPushTable = (oldFingerprint >> data->shift); // get the index in the push table

	// calculate the new fingerprint by adding the contribution of the byte pushed
	INT_64 newFingerprint = (oldFingerprint << 8) | (byte & 0xFF);
	return (newFingerprint) ^ data->pushTable[idexInPushTable];
}

inline __host__    __device__ POLY_64 update(rabinData* data, BYTE byte,
		POLY_64 oldFingerprint, byteBuffer* buffer) {

	/*
	 * the function utilizes the pop and push routines in order to
	 * push a new byte onto the sliding window, factor in its cont-
	 * ribution and remove the contribution of the byte that falls off
	 */

	oldFingerprint = popAByte(data, byte, oldFingerprint, buffer);
	//now add the new byte using the push function
	POLY_64 newFP = pushAByte(oldFingerprint, data, byte);

	return newFP;
}

#endif /* RABINFINGERPRINT_CUH_ */
