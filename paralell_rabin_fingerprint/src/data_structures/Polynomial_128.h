/**
 * Polynomial_128.cuh
 *
 * In some cases of multiplication we need to express the intermediate result
 * (the one that is still not modded) as a polynomial of a degree higher than
 * 63. In this case we can express that, using a pair of 64 bit integers where
 * one of them would represent the bits that have significance higher than 64.
 *
 *
 *
 *  Created on: May 31, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef POLYNOMIAL_128_H_
#define POLYNOMIAL_128_H_

typedef struct {
	/*
	 * Represents the bits that have significance > 63. So the least significant
	 * bit in this integer would effectively have contribution of 64 s (X^64).
	 */
	u_int64_t highBits;
	/*
	 * Those bits here are just normal with contributions ranging from 0 to 63.
	 */
	u_int64_t lowBits;
} Polynomial_128;

#endif /* POLYNOMIAL_128_H_ */
