/**
 * Buffer.cuh
 *
 * This file contains a type definition of a rather primitive cyclic buffer that
 * is used by the fingerprinting algorithm to store that contents of the window
 * of bytes being used in the computation of the fingerprint. This is useful since
 * at any time that we update the fingerprint with a new byte, in order to compute
 * the new hash efficiently (by resulting the old one), we need to know the value of
 * the byte that will be removed from the fingerprint. Therefore, we use this data
 * structure which will return this byte that we are interested in, each time we,
 * insert a new one (given that the buffer is full; and in fact this is the only
 * case in which we care about the value of this byte, since our aim is to maintain
 * a fingerprint that represents the full buffer). An important note to be made is
 * that this code can be called both from the device and from the host. This is done
 * to provide more flexibility with unit testing and debugging.
 *
 *
 *  Created on: May 31, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

#ifndef BUFFER_H_
#define BUFFER_H_

#define BUFFER_SIZE 48
typedef unsigned char BYTE;

/**
 * Quite primitive implementation of a cyclic buffer
 * to hold a window of bytes. This window is used to
 * obtain the first pushed byte on a window of size X.
 * The value is used to remove its contribution from
 * the finger print calculated.
 */
typedef struct {
	int bufptr;
	bool ifFull;
	unsigned char buf[BUFFER_SIZE];
} byteBuffer;

/**
 * Method to initialize the empty buffer and set the pointer
 * to the start of the array. Takes a pointer to the struct containing
 * the buffer variables
 *
 * @param buf pointer to the buffer
 */
__host__ __device__ void initBuffer(byteBuffer* buf);

/**
 * Resets the buffer to its initial position. Same as initBuffer(), but different
 * name for readability purposes.
 *
 * @param buf pointer to the buffer
 */
__host__ __device__ void resetBuffer(byteBuffer* buf);

/**
 *  Checks whether the particular buffer is full. A buffer is full when the bytes that
 *  have been pushed in it are more or equal to its size
 *
 * @param buf pointer to the buffer
 * @return bool signifying whether the buffer is full
 */
__host__ __device__ bool isFull(byteBuffer* buf);

/**
 * Pushes a new byte into the buffer and returns the one that was in the old position.
 * What effectively happens is that this will return the byte that was inserted first
 * in case the buffer is full.
 *
 * @param b the BYTE that is being inserted
 * @param buf a pointer to the buffer
 * @return the byte that is being removed from the buffer
 */
__host__  __device__ BYTE push(BYTE b, byteBuffer* buf);





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



inline __host__ __device__ void resetBuffer(byteBuffer* buf)
{
	initBuffer(buf);
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




#endif /* BUFFER_H_ */
