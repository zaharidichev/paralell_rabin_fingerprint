/*
 * BitOps.cuh
 *
 *  Created on: May 31, 2013
 *      Author: zahari
 */

#ifndef BITOPS_CUH_
#define BITOPS_CUH_



/*
 * Checks the value of the bit of a  at position index.
 */
inline __host__ __device__  int checkBit(u_int64_t number, u_int64_t index) {
	return (((number >> index) & 1) == 1);
}

/*
 * Returns the index of the maximum set bit. That would be the most
 * significant bit. Useful when finding out degrees of polynomials
 * that are stored in 64 bit integers.
 */
inline __host__ __device__  int getLastSetBit(u_int64_t number) {
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
