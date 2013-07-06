/**
 * ChunkingKernel.cu
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef CHUNKINGKERNEL_CU_
#define CHUNKINGKERNEL_CU_

#include "../../rabin_fingerprint/Chunker.h"
#include  "cuda_runtime.h"
#include "ResourceManagement.h"
#include "KernelStarter.h"
#include "BitFieldArray.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ int getThrID() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ void getThreadBounds(threadBounds* bounds, int dataLn, int threadsUsed, int thrID, int workPerThr) {

	bounds->start = thrID * workPerThr;

	//ACCOUTN FOR ANY LEFTOVER DATA THAT CANNOT BE DISTRIBUTED ;)
	bounds->end = (thrID == threadsUsed - 1) ? bounds->end = dataLn : bounds->start + workPerThr;

}

__global__ void findBreakPoints(rabinData* deviceRabin, BYTE* data, int dataLen, bitFieldArray results, int threadsUsed, int workPerThread, int divisor) {

	int thrID = getThrID();

	if (thrID < threadsUsed) {

		threadBounds dataBounds;

		getThreadBounds(&dataBounds, dataLen, threadsUsed, thrID, workPerThread);
		//printf("%d,%d -----> %d \n", thrID, dataBounds.start, dataBounds.end);

		/*		//defining contex for the chunker
		 chunkingContext ctx;
		 ctx.D = D_;
		 ctx.Ddash = D_DASH;
		 ctx.minThr = MIN_SIZE;
		 ctx.maxThr = MAX_SIZE;*/

		chunkDataFreeMode(deviceRabin, data, dataBounds, divisor, results, threadsUsed);
	}
}

void startCreateBreakpointsKernel(int blocksSize, int numBlocks, rabinData* deviceRabin, BYTE* deviceData, int dataLen, bitFieldArray results, int threadsUsed,
		int workPerThread, int D) {
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);


	findBreakPoints<<<numBlocks, blocksSize>>>(deviceRabin, deviceData, dataLen, results, threadsUsed, workPerThread, D);

	gpuErrchk( cudaGetLastError() );

	cudaThreadSynchronize();
}

/*int f() {
 ////////////globals
 int sizeOfData = 55574528;
 int workPerThread = 262144;

 ///////////////////////////
 cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 5242880);

 rabinData* deviceData;
 initRabinDataOnDevice(0xbfe6b8a5bf378d83, &deviceData);

 BYTE* randomHostData = allocateData(sizeOfData);
 BYTE* randomDeviceData;
 uploadDataToDevice(randomHostData, &randomDeviceData, sizeOfData);

 bool* resultsDevice;
 allocateBPArrayOnDevice(&resultsDevice, sizeOfData);

 int numberThreads = getNumNeededThreads(sizeOfData, workPerThread);

 int blocksize = 160;

 int numBlocks = numberThreads / blocksize;
 if (numberThreads % blocksize) {
 ++numBlocks;
 }
 //printf("%d %d", numBlocks, blocksize);
 findBreakPoints<<<numBlocks, blocksize>>>(deviceData, randomDeviceData, sizeOfData, resultsDevice, numberThreads, workPerThread, 512);

 CUDA_CHECK_RETURN(cudaFree(deviceData));
 CUDA_CHECK_RETURN(cudaFree(randomDeviceData));
 CUDA_CHECK_RETURN(cudaFree(resultsDevice));

 }*/

#endif /* CHUNKINGKERNEL_CU_ */
