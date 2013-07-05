/**
 * funkyFunks.h
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include <stdio.h>
#include "../../etc/helpers/Macros.h"
#include "cuda_runtime.h"
#include "../../rabin_fingerprint/RabinFingerprint.h"
#ifndef FUNKYFUNKS_H_
#define FUNKYFUNKS_H_

inline bool* allocateBPArrayOnDevice(int sizeOfData) {
	bool* breakpoints_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &breakpoints_d, sizeof(bool) * sizeOfData));
	CUDA_CHECK_RETURN(cudaMemset((void* )breakpoints_d, 0, sizeof(bool) * sizeOfData));
	return breakpoints_d;
}

inline bool* downloadBreakPointFromDevice(bool* deviceData, int size) {
	bool* rawBreakpoints = (bool*) malloc(sizeof(bool) * size);

	CUDA_CHECK_RETURN(cudaMemcpy(rawBreakpoints, deviceData, sizeof(bool) * size, cudaMemcpyDeviceToHost));
	return rawBreakpoints;
}

inline BYTE* uploadDataToDevice(BYTE* hostData, int size) {
	BYTE* dataToFingerprint_d;

	CUDA_CHECK_RETURN(cudaMalloc((void** ) &dataToFingerprint_d, sizeof(BYTE) * size));
	CUDA_CHECK_RETURN(cudaMemcpy(dataToFingerprint_d, hostData, sizeof(BYTE) * size, cudaMemcpyHostToDevice));
	return dataToFingerprint_d;

}

inline BYTE* allocateDeviceBuffer(size_t size) {
	BYTE* bufferPtr_d;
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &bufferPtr_d, sizeof(BYTE) * size))
	return bufferPtr_d;
}

inline void uploadDataToDeviceBuffer(BYTE* hostData, BYTE* devBuffer, size_t offset, size_t size) {
	hostData = hostData + offset;
	CUDA_CHECK_RETURN(cudaMemcpy(devBuffer, hostData, sizeof(BYTE) * size, cudaMemcpyHostToDevice));
}

inline rabinData* initRabinDataOnDevice(POLY_64 irrPoly) {

	rabinData hostData;
	rabinData* deviceData;
	initWindow(&hostData, irrPoly);
	CUDA_CHECK_RETURN(cudaMalloc((void** ) &deviceData, sizeof(rabinData)));
	CUDA_CHECK_RETURN(cudaMemcpy(deviceData, &hostData, sizeof(rabinData), cudaMemcpyHostToDevice));
	return deviceData;

}

inline int getNumNeededThreads(int dataLn, int minWorkPerThread) {
	//int minWorkPerThread = 262144;

	//return (dataLn % minSizePerThread != 0) ? (dataLn / minSizePerThread + 1) : dataLn / minSizePerThread;
	return (dataLn <= minWorkPerThread) ? 1 : dataLn / minWorkPerThread;
}

template<typename T> void freeCudaResource(T* resrcPtr) {
	CUDA_CHECK_RETURN(cudaFree(resrcPtr));
}

#endif /* FUNKYFUNKS_H_ */
