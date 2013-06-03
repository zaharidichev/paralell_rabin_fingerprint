/**
 * buffer.cu
 *
 *Implementation of the functions described in Buffer.cuh
 *
 *  Created on: May 29, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */



#ifndef BUFFER_CU
#define BUFFER_CU

#include "Buffer.cuh"

#define BUFFER_SIZE 48
typedef unsigned char BYTE;



 inline __host__ __device__  void initBuffer(byteBuffer* buf) {



	// cannot do that within a CUDA kernel...
	//memset(buf->buf, 0, BUFFER_SIZE);

	for (int i = 0; i < BUFFER_SIZE; ++i) {
		//  this little... thing caused me so much trouble... !
		buf->buf[i] = 0;
	}

	// setting the pointer to the beginning of the array

	buf->bufptr = 0;
	buf->ifFull = 0;
}



 inline  __device__  bool isFull(byteBuffer* buf) {
 	/*
	 * simply trusting that push will set the full toggle to
	 * true when the buffer is indeed full
	 */
	return buf->ifFull;
}

 inline __device__  unsigned char push(BYTE b, byteBuffer* buf) {

	if (++buf->bufptr >= BUFFER_SIZE) {
		/*
		 * if the buffer is full, wet the pointer to
		 * point to the first elements pushed
		 */
		buf->ifFull = true;
		buf->bufptr = 0;
	}
	// retrieves the byte that was at the place where this is pushed
	BYTE oldInPlace = buf->buf[buf->bufptr];

	buf->buf[buf->bufptr] = b; // Overwrites the position in the array
	return oldInPlace;
}

#endif /* BUFFER_CU*/
