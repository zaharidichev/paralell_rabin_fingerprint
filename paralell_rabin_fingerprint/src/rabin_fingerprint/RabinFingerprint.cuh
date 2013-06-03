/**
 * RabinFingerprint.cuh
 *
 *This file contains the signatures for a set of functions that facilitate the
 *creation and manipulation efficient rolling hashes using the Rabin's scheme.
 *This particular scheme is used in the famous sting matching algorithm proposed
 *by Micael Rabin. In our scenario we used the fingerprinting method dues to its
 *attractive properties that allow us to compute each hash in a sequence of data
 *in constant time.
 *
 *The method works as follows. If we have a sequence of bits A = {b1, b2,...bn}.
 *From this string we can construct a polynomial of degree k - 1 with coefficients
 *in GF(2) such that we have:
 *
 *A(t)  = b1^n-1 + b2^n-2 + bm;
 *
 *
 *We pick a random irreducible polynomial of degree K and create the fingerprint
 *of this sequence of bits by applying:
 *
 *F(A) = A(t) mod P(t);
 *
 *One can see that at any given point we only need to store a polynomial of degree
 *that  is no greater of the degree of P(t) (due to the modulo arithmetic). This is
 *why we can rely on representing this as a 64 bit number , given that we pick an
 *irreducible polynomial of degree that is at most 63.
 *
 *More extensive explanation of the properties of this scheme are described in the
 *relevant literature. Furthermore the scheme has been successfully used in LBFS in
 *order to compute rolling hashes over a stream of bytes. One important property
 *that needs mentioning is that in order to add one more bit to the fingerprint, we
 *simply need to shift left and mod again with our irreducible poly of choice.
 *This allows us to add entire bytes (which is ultimately the purpose).
 *
 *
 *
 *  Created on: Jun 1, 2013
 *      Author: zahari <zaharidichev@gmail.com>
 */

typedef u_int64_t POLY_64;
typedef Polynomial_128 POLY_128;

typedef struct {
	//POLY_64 fingerprint; // the current fingerprint
	//byteBuffer *byteWindow; // pointer to contents of the sliding window
	POLY_64 Irreducble_PT;
	POLY_64 T[256]; // mod lookupTable
	POLY_64 U[256]; // push lookupTable
	int shift;
} rabinData;

/**
 * Initialises the data
 * @param fingerprintData
 */
__host__ __device__ void precomputeTables(rabinData* fingerprintData);

__host__ __device__ void initWindow(rabinData* window, POLY_64 PT);

__host__     __device__ INT_64 append8(INT_64 f, INT_64 *T, BYTE m, int shift);

__host__     __device__ POLY_64 update(rabinData* data, BYTE m, POLY_64 fingerprint,
		byteBuffer* buffer);

