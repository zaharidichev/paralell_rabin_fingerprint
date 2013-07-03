/**
 * BitFieldArray.h
 *
 *  Created on: Jul 3, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "cuda_runtime.h"
#include "../../etc/helpers/Macros.h"
#include "stdio.h"
#ifndef BITFIELDARRAY_H_
#define BITFIELDARRAY_H_

typedef struct bitFieldArray {
	u_int32_t* bits;
	size_t size;
} bitFieldArray;

inline __host__ bitFieldArray createBitFieldArrayOnDevice(size_t size) {

	bitFieldArray array_h;
	array_h.size = size;
	bitFieldArray array_d;

	//uploading the structure to the device
	cudaMalloc((void**) &array_d, sizeof(bitFieldArray));
	//CUDA_CHECK_RETURN(cudaMemcpy(&array_d, &array_h, sizeof(bitFieldArray), cudaMemcpyHostToDevice));

	//now we need to allocate space for the actual results...
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &(array_d.bits), sizeof(u_int32_t) * size));
	CUDA_CHECK_RETURN(cudaMemset(array_d.bits, 0, sizeof(u_int32_t) * size));
	return array_d;
}

inline __host__ bitFieldArray downloadBitFieldArrayFromDevice(size_t size, bitFieldArray array_dev) {

	bitFieldArray array_h;
	array_h.size = size;
	array_h.bits = (u_int32_t*) malloc(sizeof(u_int32_t) * size);
	CUDA_CHECK_RETURN(cudaMemcpy(array_h.bits, array_dev.bits, sizeof(u_int32_t) * size, cudaMemcpyDeviceToHost));
	return array_h;
}

inline __device__ __host__ void setWord(int pos, u_int32_t word, bitFieldArray* array) {
	array->bits[pos] = word;
}

inline __host__ void destroyBitFieldArray(bitFieldArray* array) {
	CUDA_CHECK_RETURN(cudaFree(array->bits));
	CUDA_CHECK_RETURN(cudaFree(array));
}

inline __device__ __host__ void setBit(u_int32_t* word, int x) {
	(*word) |= 1 << x;
}

inline __device__ __host__ void setReverseBit(u_int32_t* word, int x) {
	int index = 31 - x;
	setBit(word, index);
}

inline __host__ bool getBit(size_t pos, bitFieldArray* array) {
	int posInArray = (pos) / 32;
	int bitIndex = 31 - (pos % 32);
	return (array->bits[posInArray] & (1 << bitIndex));

}

inline __host__ void printWordInBinary(u_int32_t a) {

	int bits[32];
	int i;
	for (i = 0; i < 32; i++) {
		bits[31 - i] = (a >> i) & 1; //store the ith bit in b[i]
	}

	for (int i = 0; i < 32; ++i) {

		printf("%d", bits[i]);
	}
}

#endif /* BITFIELDARRAY_H_ */
