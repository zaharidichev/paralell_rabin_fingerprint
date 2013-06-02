/*
 *
 * This is a collection of functions that are useful when manipulating 64 bit integer
 * numbers on bit level.
 *
 * BitOps.cuh
 *
 *  Created on: May 31, 2013
 *      Author: zahari
 */

#ifndef BITOPS_CUH_
#define BITOPS_CUH_

/**
 * Checks the value of the bit of a specified index for a 64 bit number
 *
 * @param number the integer containing the bits
 * @param index the index of the bit that we are interested in
 * @return 1 if the bit is set, 0 if it is not
 */
inline __host__ __device__ int checkBit(u_int64_t number, u_int64_t index) {
	return (((number >> index) & 1) == 1);
}

/**
 * Returns the index of the maximum set bit. That would be the most
 * significant bit. Useful when finding out degrees of polynomials
 * that are stored in 64 bit integers.
 *
 *
 * @param number the 64 bit number
 * @return the index of the bit
 */
inline __host__ __device__ int getLastSetBit(u_int64_t number) {
	int i = 64 - 1;
	while (i >= 0) {
		// loop and use the check bit function for readability
		if (checkBit(number, i))
			return i;
		i--;
	}
	return -1;
}

#endif /* BITOPS_CUH_ */
