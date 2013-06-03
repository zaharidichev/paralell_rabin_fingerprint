#include <stdio.h>
#include <stdlib.h>
#include "data_structures/Buffer.cu"
#include "math/PolyMath.cu"
#include "rabin_fingerprint/RabinFingerprint.cu"
#include<cuda_runtime.h>

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 * kernel that creates fingeprrints for a sliding window of 48b over 256 long stream
 */ __global__ void RabinExample(rabinData* deviceRabin, BYTE* data,
		POLY_64* results) {

	// create and initialize the local window buffer
	byteBuffer b;
	initBuffer(&b);

	POLY_64 fingerprint = 0;

	for (int i = 0; i < 256; ++i) {
		// iterate through the data to be hashed and write fingerprints to the result buffer
		fingerprint = update(deviceRabin, data[i], fingerprint, &b);
		results[i] = fingerprint;
	}
}

/**
 * main function that just launced the kernel and takes care of book kepping
 */


void fingerprint256Bytes(POLY_64* results, BYTE*data, POLY_64 irreduciblePoly) {

	// variables for our data

	//data
	rabinData hostData; // host_window
	rabinData* deviceData; // device_window

	// we first init the window on the host (ideally we should copy that to constant mem)
	initWindow(&hostData, irreduciblePoly);

	//allocate device memory for the fingerprinting data
	cudaMalloc((void**) &deviceData, sizeof(rabinData));

	//copy the data to device
	CUDA_CHECK_RETURN(
			cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));

	// allocate space for the data that we need to fingerprint
	BYTE* dataToFingerprint_d;
	// init the data to some values

	// allocate memory on the host and copy the data
	cudaMalloc((void**) &dataToFingerprint_d, sizeof(BYTE) * 256);
	CUDA_CHECK_RETURN(
			cudaMemcpy(dataToFingerprint_d, data, sizeof(BYTE) * 256, cudaMemcpyHostToDevice));

	// now we also need some space for the results
	POLY_64* results_d;
	cudaMalloc((void**) &results_d, sizeof(POLY_64) * 256);

	// call the kernel that will create the fingerprints

	RabinExample<<<1, 1>>>(deviceData, dataToFingerprint_d, results_d);


	//copy back into our supplied data
	CUDA_CHECK_RETURN(
			cudaMemcpy(results, results_d, sizeof(POLY_64) * 256, cudaMemcpyDeviceToHost));

	// free all the memory alocated on the card
	CUDA_CHECK_RETURN(cudaFree(deviceData));
	CUDA_CHECK_RETURN(cudaFree(dataToFingerprint_d));
	CUDA_CHECK_RETURN(cudaFree(results_d));

}

