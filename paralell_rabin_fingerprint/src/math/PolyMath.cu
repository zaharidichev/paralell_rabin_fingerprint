

#include "PolyMath.cuh"
#include "BitOps.cu"
#include "stdio.h"
#ifndef POLYMATH_CU
#define POLYMATH_CU




inline __host__ __device__ POLY_64 mod(POLY_64 x, POLY_64 y) {

	int degreeOfX = degree(x); // get degree of x
	int degreeOfY = degree(y); // get degree of y

	for (int i = degreeOfX - degreeOfY; i >= 0; i--) {
		if (checkBit(x, (i + degreeOfY))) {
			// if the bit is degree contribution
			// shift y by the difference between the degrees (synthetic division algorhithm)
			uint64_t shiftedByDiference = y << i;
			//xor X with the shifted result
			x = x ^ shiftedByDiference;
		}

	}
	return x;
}

inline __host__ __device__ int degree(POLY_64 p) {
	return getLastSetBit(p); // get the last set bit (the one with the highest significance)
}






inline __host__ __device__ POLY_128 mult_128(POLY_64 x, POLY_64 y) {
	//defining high and low bits of 128 poly
	POLY_64 highBits = 0;
	POLY_64 lowBits = 0;

	if ((x & POLY_64(1)) != 0) {
		lowBits = y;
	}
	for (int i = 1; i < 64; i++) {
		/*
		 * first check whether the bit is set at all
		 */
		if ((x & (INT_64(1) << i)) != 0) {

			/*
			 * very efficient way of multiplication by shifting
			 * and then XOR-ring, rather than iterating through all
			 * all the terms. This trick is used in the source code
			 * of LBFS. Remeber that when it comes to multiplication
			 * we have term by term addition and that in GF(2) addition
			 * can be expresses as XOR operation for every term
			 */
			highBits ^= y >> (64 - i);
			lowBits ^= y << i;
		}
	}

	//constructing polynomial of higher than 63 degree
	POLY_128 result;
	result.highBits = highBits;
	result.lowBits = lowBits;

	return result;
}

inline __host__ __device__ POLY_64 polyModmult(POLY_64 x, POLY_64 y, POLY_64 d) {

	POLY_128 product = mult_128(x, y); // we first multiply the two polys
	return mod_128(product, d); // and then return the result of modding the product by d
}



inline __host__ __device__ POLY_64 mod_128(POLY_128 x, POLY_64 d) {
	INT_64 highBits = x.highBits;
	INT_64 lowBits = x.lowBits;

	POLY_64 k = degree(d);
	d <<= 63 - k;

	if (highBits != 0) {
		if ((highBits & INT_64(0x8000000000000000)) != 0) {
			highBits ^= d;
		}
		for (int i = 62; i >= 0; i--) {
			if ((highBits & (INT_64(1)) << i) != 0) {
				highBits ^= d >> (63 - i);
				lowBits ^= d << (i + 1);
			}
		}
	}
	for (int i = 63; i >= k; i--) {
		if ((lowBits & INT_64(1) << i) != 0)
			lowBits ^= d >> (63 - i);
	}
	return lowBits;
}

inline __host__  void printPolyAsEquationString(POLY_64 poly) {
	/*
	 * we do not need to go through all the bits one by one since we can just
	 * get the highest set one (designating the degree of our polynomial and
	 * use it as a starting point)
	 */
	int polyDegree = getLastSetBit(poly); // get degree of polynomial
	for (int bitIndex = polyDegree; bitIndex >= 0; --bitIndex) {
		/*
		 * loop through all the bits starting from the most significant
		 * and moving right
		 */
		if (checkBit(poly, bitIndex)) {
			// if the bit is set
			if (bitIndex != polyDegree) {
				//if we are not at the beginning. print +
				printf(" + ");

			}
			if (bitIndex == 0) {
				/*
				 * if this is the LSB and it is set, we have
				 * a last element of 1
				 */
				printf("1");

			} else {
				// else just print X^ the power
				printf("x^%u", bitIndex);


			}

		}
	}
}

inline __host__ void printPolyAsHEXString(POLY_64 p) {
	printf("%016llX", p);


}

inline __host__ void printPolyAsBinaryString(POLY_64 a) {

	int bits[64];
	int i;
	for (i = 0; i < 64; i++) {
		bits[63 - i] = (a >> i) & 1; //store the ith bit in b[i]
	}

	for (int i = 0; i < 64; ++i) {

		printf("%d", bits[i]);
	}
}
#endif
