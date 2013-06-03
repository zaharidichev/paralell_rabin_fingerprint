/*
 * RabinFingerprint.cuh
 *
 *  Created on: May 31, 2013
 *      Author: zahari
 */

#ifndef RABINFINGERPRINT_CUH_
#define RABINFINGERPRINT_CUH_
#include "../math/BitOps.cu"
#include "../math/PolyMath.cu"
#include "../data_structures/Buffer.cu"
#include "RabinFingerprint.cuh"




inline __host__ __device__ INT_64 append8(INT_64 f, INT_64 *T, BYTE m, int shift) {
	return ((f << 8) | (m & 0xff)) ^ T[(int) (f >> shift)];
}



inline __host__ __device__  void precomputeTables(rabinData* fingerprintData) {

	//fingerprintData->fingerprint = 0;
	int fDegree = degree(fingerprintData->Irreducble_PT);
	fingerprintData->shift = fDegree - 8;
	long T1 = mod((INT_64(1) << fDegree), fingerprintData->Irreducble_PT);
	for (INT_64 j = 0; j < 256; j++) {
		// computing the T table
		fingerprintData->T[(int) j] = (polyModmult(j, T1,
				fingerprintData->Irreducble_PT) | (j << fDegree));
		//printPolyINHEX(fingerprintData->T[(int) j]);
		//printf(" ");
	}
	//printf("\n");
	INT_64 sizeshift = 1;
	for (INT_64 i = 1; i < WIN_SIZE; i++)
		sizeshift = append8(sizeshift, fingerprintData->T, (BYTE) 0,
				fingerprintData->shift);
	for (INT_64 i = 0; i < 256; i++) {
		fingerprintData->U[i] = polyModmult(i, sizeshift, fingerprintData->Irreducble_PT);
		//printPolyINHEX(fingerprintData->U[(int) i]);
		//printf(" ");
	}
}


inline __host__ __device__ void initWindow(rabinData* window,POLY_64 PT) {
	window->Irreducble_PT = PT;
	precomputeTables(window);
}


inline __host__ __device__ POLY_64 update (rabinData* data,BYTE m, POLY_64 fingerprint, byteBuffer* buffer ) {

   	BYTE old = push( m, buffer);
   	//printf("%d", ((int)old)  & 0xFF);
   	//printf("\n");

   	   fingerprint = fingerprint^ data->U[old & 0xff];
       POLY_64 newFP = append8(fingerprint,data->T, m, data->shift);
       return newFP;
   }


#endif /* RABINFINGERPRINT_CUH_ */
