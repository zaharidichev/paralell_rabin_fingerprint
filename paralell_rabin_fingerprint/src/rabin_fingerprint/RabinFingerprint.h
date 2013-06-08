/**
 * RabinFingerprint.cuh
 *
 *This file contains the signatures for a set of functions that facilitate the
 *creation and manipulation efficient rolling hashes using the Rabin's scheme.
 *This particular scheme is used in the famous sting matching algorithm proposed
 *by Micael Rabin. In our scenario we used the fingerprinting method dues to its
 *attractive properties that allow us to compute each hash in a sequence of data
 *in constant time.
 *
 *The method works as follows. If we have a sequence of bits A = {b1, b2,...bn}.
 *From this string we can construct a polynomial of degree k - 1 with coefficients
 *in GF(2) such that we have:
 *
 *A(t)  = b1^n-1 + b2^n-2 + bm;
 *
 *
 *We pick a random irreducible polynomial of degree K and create the fingerprint
 *of this sequence of bits by applying:
 *
 *F(A) = A(t) mod P(t);
 *
 *One can see that at any given point we only need to store a polynomial of degree
 *that  is no greater of the degree of P(t) (due to the modulo arithmetic). This is
 *why we can rely on representing this as a 64 bit number , given that we pick an
 *irreducible polynomial of degree that is at most 63.
 *
 *More extensive explanation of the properties of this scheme are described in the
 *relevant literature. Furthermore the scheme has been successfully used in LBFS in
 *order to compute rolling hashes over a stream of bytes. One important property
 *that needs mentioning is that in order to add one more bit to the fingerprint, we
 *simply need to shift left and mod again with our irreducible poly of choice.
 *This allows us to add entire bytes just by shifting left by 8.
 *
 *
 *
 *  Created on: Jun 1, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */


#ifndef RABINFINGERPRINT_H
#define RABINFINGERPRINT_H

#include "../data_structures/Buffer.h"
#include "../math/PolyMath.h"
#include "RabinData.h"



/**
 * Initializes the fingerprint data by setting all the values in the struct supplied
 *
 * @param window the struct that holds the data for the fingerprint
 * @param PT the irreducible polynomial that will be used for modding the fingerprint
 */
__host__  void initWindow(rabinData* window, POLY_64 PT);

/**
 * Precomputes the results of pushing and popping bytes so that we do not have to
 * do it every single type we update the fingerprint. This trick is borrowed from
 * the LBFS implementation of sliding window fingerprinting.
 *
 * @param fingerprintData the struct containing the data for the fingerprint
 */
__host__ void precomputeTables(rabinData* fingerprintData);

/**
 * By pushing a new byte onto the window, we effectivelly remove the one that falls
 * outside of the window (given that the window is full). This method does exactly
 * that and in addition to that it removes the contribution of the byte that fell
 * out of the window from the fingerprint so the new byte's contribution can be added.
 *
 * @param data the struct representing the current state of the fingerprint
 * @param byte the byte that needs to be pushed onto the buffer
 * @param oldFingerprint the old fingerprint that will be updated
 * @param buffer the buffer containing the contents of the sliding window
 * @return an updated fingerprint with the contribution of the byte that fell out
 * removed
 */
__host__  __device__ POLY_64 popAByte(rabinData* data, BYTE byte,
		POLY_64 oldFingerprint, byteBuffer* buffer);

/**
 * This function adds the contribution of a new byte to the fingerprint. Note that
 * the function does not manipulate the actual contents of the buffer that holds the
 * bytes. This is the responsibility of the popAByte() one. This method just deals
 * with performing the bit manipulation to the fingerprint.
 *
 * @param oldFingerprint the fingerprint that will be updated
 * @param data the auxiliary data such as push and pop tables
 * @param byte the byte that needs to be added to the fingerprint
 * @return the updated fingerprint
 */
__host__       __device__ INT_64 pushAByte(INT_64 oldFingerprint, rabinData* data,
		BYTE byte);

/**
 * This method makes use of both popAByte and pushAByte in order to abstract away the
 * process of updating the fingerprint. All we need is the new byte and the auxiliary
 * data for the fingerprint
 *
 * @param data struct containing irreducible poly, push tables and all the rest
 * @param m the byte that needs to be pushed
 * @param fingerprint the 64 bit number that represents the fingerprint
 * @param buffer the buffer that holds the current contents of the window
 * @return the updated fingerprint
 */
__host__ __device__ POLY_64 update(rabinData* data, BYTE m, POLY_64 fingerprint,
		byteBuffer* buffer);



inline  __host__  void initWindow(rabinData* window, POLY_64 PT) {
	window->Irreducble_PT = PT; // set the internal variable to the irreducible poly
	precomputeTables(window);
}

 __host__ inline  void precomputeTables(rabinData* fingerprintData) {

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

#endif /* RABINFINGERPRINT_H */
