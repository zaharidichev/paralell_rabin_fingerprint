/**
 * mainExample.cpp
 *
 * This file shows an example of how the system can be used
 *
 *  Created on: Jul 12, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "../chunking_system/kernel_wrapper/GPUChunker.h"
#include "../chunking_system/GPU_code/BitFieldArray.h"
#include "../chunking_system/IO_tools/FileReader.h"

typedef unsigned char BYTE;

int main() {

	size_t sizeOfDataToFingerprint = 737233947; // size of the file to be chunked in bytes
	size_t minSize = 32768; // the minimum chunk size
	size_t maxSize = 65536; // the maximum chunk size
	int primaryDiv = 512; // the primary divisor
	int secondaryDiv = 256; // the secondary divisor
	POLY_64 irr_Poly = 0xbfe6b8a5bf378d83; // the irreducible polynomial

	FileReader exampleFile("/home/zahari/Desktop/2.6_kernels_merged.dat"); // create a file reader

	// create a GPU chunker
	GPUChunker chunker = GPUChunker(primaryDiv, secondaryDiv, 0xbfe6b8a5bf378d83, minSize, maxSize);
	// now we chunk and retrieve the resulting pieces
	std::vector<boost::shared_ptr<Chunk> > results = chunker.chunkFileFromDisk(exampleFile, sizeOfDataToFingerprint, CONTINUOUS);

	// we can simply iterate through the vector and print the chunks
	for (std::vector<boost::shared_ptr<Chunk> >::iterator it = results.begin(); it != results.end(); ++it) {
		std::cout << *(*it).get() << std::endl;
	}


   //------------------ the output should look like this ------------------------ //

/*      Start         End      Size                SHA-1 HASH
 * --------------------------------------------------------------------------------
 * |  [400014626 - 400047976] [33350] [da96949c4b8ffc1cc681d0bbe359cfa9e5bb6970]  |
 * |  [400047976 - 400081983] [34007] [4c1849eb8c6ad1704d770ec1e779f0e16c661c70]  |
 * |  [400081983 - 400115161] [33178] [6b8e78f0ff0a74cabec66621474cb8d821e5ad23]  |
 * |  [400115161 - 400148196] [33035] [e81eede93a5c8e51b5107c2c5e10d9e4827362bf]  |
 * |  [400148196 - 400181099] [32903] [f1eff6a30cea0f127dfb9968db988e14dbde9c43]  |
 * |  [400181099 - 400214303] [33204] [bcff30240844fffe470e3d69d0763d28af1f2368]  |
 * |  [400214303 - 400247198] [32895] [efa54b40210e96e1b1a21bd1c701e2b779abeaec]  |
 * |  [400247198 - 400280022] [32824] [86925fc0bc58eb144356eeb45ff88a70ac57f65c]  |
 * |  [400280022 - 400313125] [33103] [0e572fa15a8b467eff83ba7b50071b09d821325f]  |
 * --------------------------------------------------------------------------------
 */


}
