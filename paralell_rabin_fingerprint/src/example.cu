/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "/usr/local/cuda-5.0/samples/0_Simple/simplePrintf/cuPrintf.cu"
#include "data_structures/Buffer.cu"
#include "math/PolyMath.cu"
#include "rabin_fingerprint/RabinFingerprint.cu"
#include<cuda_runtime.h>



__device__ int  cprintF(const char *,...) {
	cup
}

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





__device__ void caller(rabinData* window) {

	u_int64_t x = 551;
	u_int64_t y = 287;

	//mult(x,y);

	//printPolyAsHEXString(mult(x,y));

	//cuPrintf("%d", polyModmult(0xbfe6b8a5bf378d83, 0xbfe6b8a5bf378d83, 89));



}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */__global__ void bitreverse(rabinData* deviceRabin, BYTE* data, POLY_64* results) {

	 byteBuffer b;
	 initBuffer(&b);

	 POLY_64 fingerprint = 0;

	 for (int i = 0; i < 256; ++i) {
		 fingerprint =  update(deviceRabin, data[i], fingerprint, &b);
		 //cuPrintf("%d ", fingerprint);
		 results[i] = fingerprint;
	}
}


 __global__ void testPrint() {


		getPolyAsEquationString(0xbfe6b8a5bf378d83, &cuPrintf);


}



/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */



int main(void) {


	 rabinData hostData; // host_window
	 rabinData* deviceData; // device_window

	 initWindow(&hostData,0xbfe6b8a5bf378d83 ); // init the window on the host

	//allocate device memory for the fingerprinting data
    cudaMalloc((void**)&deviceData, sizeof(rabinData));
    //copy i

    //copy the data to device
	CUDA_CHECK_RETURN(cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));


	BYTE dataToFingerprint_h[256];
	BYTE* dataToFingerprint_d;

	for (int i = 0; i < 256; ++i) {
		dataToFingerprint_h[i] = (BYTE)i;
	}
    cudaMalloc((void**)&dataToFingerprint_d, sizeof(BYTE) * 256);
	CUDA_CHECK_RETURN(cudaMemcpy(dataToFingerprint_d, &dataToFingerprint_h, sizeof(BYTE) * 256, cudaMemcpyHostToDevice));

	POLY_64 restults_H[256];
    POLY_64* results_d;
    cudaMalloc((void**)&results_d, sizeof(POLY_64) * 256);






	cudaPrintfInit();
	//bitreverse<<<1, 1>>>(deviceData, dataToFingerprint_d, results_d);

	testPrint<<<1, 1>>>();


	cudaPrintfDisplay(stdout, false);
	cudaPrintfEnd();

	CUDA_CHECK_RETURN(cudaMemcpy(&restults_H, results_d, sizeof(POLY_64) * 256, cudaMemcpyDeviceToHost));




	printf("Hello from Host: ");
	getPolyAsEquationString(0xbfe6b8a5bf378d83, &printf);

	for (int i = 0; i < 256; ++i) {

		//printPolyAsHEXString(restults_H[i]);
		//printf(" ");

	}

	CUDA_CHECK_RETURN(cudaFree(deviceData));
	CUDA_CHECK_RETURN(cudaFree(dataToFingerprint_d));
	CUDA_CHECK_RETURN(cudaFree(results_d));
	return 0;
}
