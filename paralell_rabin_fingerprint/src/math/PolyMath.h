/**
 * PolyMath.cuh
 *
 *Contains a set of functions that perform operations on polynomials in GF(2),
 *that are expressed as 64 bit integers. It is important to note that integers
 *and their bit representation is good match for GF(2) since the set contains
 *only two elements {0,1};
 *
 *For more information regarding arithmetic in this field, take a look at:
 *
 *https://engineering.purdue.edu/kak/compsec/NewLectures/Lecture6.pdf
 *
 *  Created on: May 30, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef POLYMATH_H_
#define POLYMATH_H_
#include "../data_structures/Polynomial_128.h"
#include "../etc/DedupDefines.h"
#include "stdint.h"
#include "BitOps.h"
#include "stdio.h"

typedef Polynomial_128 POLY_128;

/**
 * This function returns the degree of the polynomial that is represented by the 64 bit integer
 *
 * @param p the 64 bit representaio nof the polynomial
 * @return the degree
 */
inline __host__ __device__ int degree(POLY_64 p) {
	return getLastSetBit(p); // get the last set bit (the one with the highest significance)
}

/**
 * Performs X % Y for binary polynomials represented as bit registers.
 *
 * @param x the first polynomial
 * @param y the second polynomial
 * @return the remainder
 */
inline __host__   __device__ POLY_64 mod(POLY_64 x, POLY_64 y) {

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

/**
 * Multiplies two polynomial of degree up to 63. The result of that is a
 * polynomial that can be of higher degree, therefore the structure representing
 * it is a C struct of two 64 bit integers, the first one indicating the lower
 * exponents and second one the higher ones.
 *
 * @param x the first polynomial
 * @param y the second polynomial
 * @return the product
 */
inline __host__   __device__ POLY_128 mult_128(POLY_64 x, POLY_64 y) {
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

/**
 * Performs modular reduction of a polynomial of degree up to 127 by a polynomial of
 * degree up to 63. The higher polynomial is represented as a struct of two 64 bit integers.
 * The result is a 64 bit representation of the resulting polynomial. We do not need more
 * than that since X % Y will never be higher than Y.
 *
 * @param x the large polynomial
 * @param d the smaller polynomial
 * @return the result
 */
inline __host__   __device__ POLY_64 mod_128(POLY_128 x, POLY_64 d) {
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

/**
 * This function uses the above ones to perform modular multiplication. The two polynomials
 * x and y are multiplied and the result is modded by the D parameter, which will always
 * result in a polynomial of degree 63 or lower.
 *
 * @param x the first poly
 * @param y the second poly
 * @param d the divisor poly
 * @return the result as a 64 bit integer representing a binary polynomial
 */
inline __host__   __device__ POLY_64 polyModmult(POLY_64 x, POLY_64 y, POLY_64 d) {

	POLY_128 product = mult_128(x, y); // we first multiply the two polys
	return mod_128(product, d); // and then return the result of modding the product by d
}

/**
 *
 * This function prints a 64 bit integer as a polynomial equation
 *
 * @param poly the binary polynomial represents in a 64 bit integer
 */
inline __host__ void printPolyAsEquationString(POLY_64 poly) {
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

/**
 * The function prints a polynomial as a hex sequence
 * @param p the binary polynomial as an integer
 */
inline __host__ void printPolyAsHEXString(POLY_64 p) {
	printf("%016llX", p);

}

/**
 * Prints the exact  binary pattern that is contained in the 64 bits of the integer
 * that represents the polynomial
 *
 * @param a the 64 bit integer
 */
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

#endif /* POLYMATH_H_ */
