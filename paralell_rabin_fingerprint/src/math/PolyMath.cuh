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
 *      Author: zahari
 */

#ifndef POLYMATH_CUH_
#define POLYMATH_CUH_
#include "../etc/DedupDefines.h"
#include "../data_structures/Polynomial_128.cuh"
#include "/usr/local/cuda-5.0/samples/0_Simple/simplePrintf/cuPrintf.cuh"




/*
 * This is a function pointer to our printing function.
 * We need to do that since on older cuda architectures
 * the use of printf() in device code is not possible
 * and alternatively cuPrintf is used;
 */


//typedef (*print_func)(const char *,...) print
typedef u_int64_t POLY_64;
typedef  Polynomial_128 POLY_128;




/**
 * Performs X % Y for two polynomials in GF(2) that have a maximum degree of 63.
 * In this case our modding requires just a
 *
 *
 * @param x the first polynomial
 * @param y the second polynomial
 * @return the result as a 64 bit unsigned number
 */

__host__ __device__ POLY_64 mod(POLY_64 x, POLY_64 y);

/**
 * Uses the last set bit function from BitOps_64.h in order to return the
 * degree of a particular polynomial that is expressed as a 64 bit number
 *
 * @param p the polynomial that we need to expect
 * @return the degree of the supplied polynomial
 */

__host__ __device__ int degree(POLY_64 p);

/*-------------------------------------------------------
 * LBFS way of doing things :)
 -------------------------------------------------------*/

/**
 * This function is used when multiplying polynomials that might result
 * in another polynomial of a degree higher than 63. In this case the
 * result is returned as a struct containing to 64 bit numbers representing
 * the polynomial
 *
 * @param x the first polynomial (64)
 * @param y the second polynomial (64)
 * @return a struct of 2 64 bit numbers that represent a polynomial of degree
 * that is potentially higher than 63
 */

__host__ __device__ POLY_128 mult_128(POLY_64 x, POLY_64 y);

/**
 * This function is used to multiply two polynomials of degrees up to 63, which,
 * can potentially produce a result of degree higher than what a 64 bit number
 * can handle. We are generally interested in modding this result by some other
 * polynomial which cannot result in degree higher than 63. This function aids
 * in doing so and returns us the result of (x*y)%d where all variables are
 * polynomials that can be represented in 64 bit numbers without overflowing.
 *
 * @param x the first poly
 * @param y the second poly
 * @param d the modder
 * @return the result of (x*y)%d
 */

__host__ __device__ POLY_64 polyModmult(POLY_64 x, POLY_64 y, POLY_64 d);

/**
 * This function is the same as the normal % but in this case the polynomial
 * that will me modded, can be of degree that needs to be represented using two
 * 64 bit numbers instead of one.
 *
 * @param x a 128bit representation of a polynomial in GF(2)
 * @param 64 bit representation of a polynomial in GF(2)
 * @return we only need to return a 64 bit result since we know that the result of
 * x(128) % d(64) cannot be more than 64 bit large.
 */

__host__ __device__ POLY_64 mod_128(POLY_128 x, POLY_64 d);

/**
 * Prints a Polynomial in the form of X^63 + X^7 + X^4 + 1. Useful for debugging/
 *
 * @param poly the polynomial that needs to be printed
 */
__host__ void printPolyAsEquationString(POLY_64 poly);

/**
 * Prints a 64 bit number, representing a polynomial as a HEX string
 *
 * @param p  the integer to be printed
 */
 __host__ void printPolyAsHEXString(POLY_64 p);

/**
 * Prints the bit pattern that is contained in a 64 bit integer (unsigned)
 * @param a
 */
__host__  void printPolyAsBinaryString(POLY_64 a);

#endif /* POLYMATH_CUH_ */
