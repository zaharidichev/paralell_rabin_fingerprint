/**
 * BitFieldArray.h
 *
 *This file provides the functions that are needed to manipulate a highly efficient
 *data structure, which purpose is to store bool values. The way the structure works
 *is that is stores consecutive 32bit figures in memory where each bit represents
 *a bool values that can be either true or not. The functions provided can manipulate
 *this data structure and set individual bits in a transparent to the user fashion by
 *using bitwise arithmetic.
 *
 *  Created on: Jul 3, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef BITFIELDARRAY_H_
#define BITFIELDARRAY_H_
#include "cuda_runtime.h"
#include "../../etc/helpers/Macros.h"
#include "stdio.h"
#include "boost/shared_ptr.hpp"

using namespace boost;
typedef u_int32_t* bitFieldArray;
typedef u_int32_t word32;

/**
 * Calculates the length of a bitField array that is needed for a piece of data of a specified length
 * @param dataLn the length of the data in bytes
 * @return the size of the array (or how many 32 bit figures are needed to fit breakpoints for that data)
 */

inline __host__ size_t getSizeOfBitArray(size_t dataLn) {
	return (dataLn % sizeof(word32) == 0) ? dataLn / sizeof(word32) : (dataLn / sizeof(word32)) + 1;

}
/**
 * This function flushes the bitfield data structure by setting all the bits within
 * it to 0;
 * @param array pointer to the bitfield array
 * @param size the size of the structure (the size in 32 bit figures)
 */
inline __host__ void flushBitfieldFufferOnDevice(bitFieldArray array, size_t size) {

	CUDA_CHECK_RETURN(cudaMemset(array, 0, sizeof(word32) * size));
}

/**
 * The function abstract away the allocation of a an array of bits on the device.
 * The size is the number of 32bit figures that are needed.
 * @param size size of the structure
 * @return pointer to the structure on the device
 */
inline __host__ bitFieldArray createBitFieldArrayOnDevice(size_t size) {

	bitFieldArray array_d;

	//now we need to allocate space for the actual results...
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &array_d, sizeof(word32) * size));
	CUDA_CHECK_RETURN(cudaMemset(array_d, 0, sizeof(word32) * size));
	return array_d;
}

/**
 * This function is used to download the bit field array from the device to the host.
 *
 * @param size the size of the array
 * @param array_dev the a pointer to the array that is located on the GPU
 * @return a pointer to the copy of the array on the device
 */
inline __host__ bitFieldArray downloadBitFieldArrayFromDevice(size_t size, bitFieldArray array_dev) {

	bitFieldArray array_h;
	array_h = (word32*) malloc(sizeof(word32) * size);

	CUDA_CHECK_RETURN(cudaMemcpy(array_h, array_dev, sizeof(word32) * size, cudaMemcpyDeviceToHost));
	return array_h;
}

/**
 * Sets a 32 bit word in the array.
 * @param pos pos in the array
 * @param word the word to be placed
 * @param array a pointer to the array
 */
inline __device__ __host__ void setWord(int pos, word32 word, bitFieldArray array) {
	array[pos] = word;
}

/**
 * Frees the memory on the device
 * @param array a pointer to the array
 */
inline __host__ void destroyBitFieldArray(bitFieldArray array) {
	CUDA_CHECK_RETURN(cudaFree(array));
}

/**
 * Sets the bit of a particular word given an index (counting from the least significant bit )
 * @param word the 32bit word
 * @param x the index
 */
inline __device__ __host__ void setBit(word32* word, int x) {
	(*word) |= 1 << x;
}

/**
 * Sets the bit of a particular word given an index (counting from the most significant bit )
 * @param word the 32bit word
 * @param x the index
 */

inline __device__ __host__ void setReverseBit(word32* word, int x) {
	int index = 31 - x;
	setBit(word, index);
}

/**
 * Retrieves the value of a bit (bool) on a specified index in the array in a transparent fashion
 * @param pos the position in the array
 * @param array the pointer to the array
 * @return the value of the bit
 */
inline __device__ __host__ bool getBit(size_t pos, bitFieldArray array) {
	size_t sizeOfword = sizeof(word32) * 8;
	int posInArray = (pos) / sizeOfword;
	int bitIndex = (sizeOfword - 1) - (pos % sizeOfword);
	return (array[posInArray] & (1 << bitIndex));

}

/**
 *Prints the binary patter that represents a particular word of the array
 * @param a the word
 */
inline __host__ void printWordInBinary(word32 a) {

#define WORD_SIZE 32
	int bits[WORD_SIZE];
	int i;
	for (i = 0; i < WORD_SIZE; i++) {
		bits[(WORD_SIZE - 1) - i] = (a >> i) & 1; //store the ith bit in b[i]
	}

	for (int i = 0; i < WORD_SIZE; ++i) {

		printf("%d", bits[i]);
	}
}

#endif /* BITFIELDARRAY_H_ */
