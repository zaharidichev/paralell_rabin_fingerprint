/**
 * mainTest.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "GPUChunker.h"
#include "../GPU_code/BitFieldArray.h"
#include "../IO_tools/FileReader.h"
#include <ctime>

typedef unsigned char BYTE;

/*
 bool compareChunks(std::vector<shared_ptr<Chunk> > &left, std::vector<shared_ptr<Chunk> > &right) {

 int size = left.size();

 for (int var = 0; var < size; ++var) {
 shared_ptr<Chunk> ch1 = left[var];
 shared_ptr<Chunk> ch2 = right[var];

 std::cout << *ch1.get() << " " << *ch2.get() << std::endl;

 if (!(ch1.get()->getStart() == ch2.get()->getStart()) || !(ch1.get()->getEnd() == ch2.get()->getEnd())) {
 return false;
 }

 assert(ch1.get()->getEnd() == ch2.get()->getEnd());

 for (int var_2 = 0; var_2 < 20; ++var_2) {
 if (!(ch1.get()->getHash().get()[var_2] == ch2.get()->getHash().get()[var_2])) {
 return false;
 }
 }

 }
 return true;
 }

 int testDifferentBuffers(size_t sizeBuffer, size_t min) {

 size_t sizeOfDataToFingerprint = 2032221073;
 size_t minSize = min;
 size_t maxSize = 131072;
 int primaryDiv = 512;
 int secondaryDiv = 256;
 POLY_64 irr_Poly = 0xbfe6b8a5bf378d83;

 FileReader infile1("/home/zahari/Desktop/2.6_kernels_merged.dat");

 GPUChunker chunker = GPUChunker(primaryDiv, secondaryDiv, 0xbfe6b8a5bf378d83, minSize, maxSize);

 using namespace std;
 clock_t begin = clock();

 std::vector<shared_ptr<Chunk> > chunks_segmeted = chunker.chunkFileFromDisk(infile1, sizeOfDataToFingerprint, SEGMENTED, sizeBuffer);

 clock_t end = clock();
 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
 printf("%d %.2lf\n", min, elapsed_secs);

 }

 int testDifferentBuffers_cont(size_t sizeBuffer, size_t min) {

 size_t sizeOfDataToFingerprint = 2032221073;
 size_t minSize = min;


 size_t maxSize = 131072;
 int primaryDiv = 512;
 int secondaryDiv = 256;
 POLY_64 irr_Poly = 0xbfe6b8a5bf378d83;

 FileReader infile1("/home/zahari/Desktop/2.6_kernels_merged.dat");

 GPUChunker chunker = GPUChunker(primaryDiv, secondaryDiv, 0xbfe6b8a5bf378d83, minSize, maxSize);

 using namespace std;
 clock_t begin = clock();

 std::vector<shared_ptr<Chunk> > chunks_segmeted = chunker.chunkFileFromDisk(infile1, sizeOfDataToFingerprint, CONTINUOUS, sizeBuffer);

 clock_t end = clock();
 double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
 printf("%d %.2lf\n", sizeBuffer, elapsed_secs);

 }
 */

int main() {

	size_t sizeOfDataToFingerprint = 737233947;
	size_t minSize = 32768;

	size_t maxSize = 65536;
	int primaryDiv = 512;
	int secondaryDiv = 256;
	POLY_64 irr_Poly = 0xbfe6b8a5bf378d83;

	FileReader infile1("/home/zahari/Desktop/2.6_kernels_merged.dat");

	GPUChunker chunker = GPUChunker(primaryDiv, secondaryDiv, 0xbfe6b8a5bf378d83, minSize, maxSize);
	std::vector<boost::shared_ptr<Chunk> > results = chunker.chunkFileFromDisk(infile1, sizeOfDataToFingerprint, CONTINUOUS);

	for(std::vector<boost::shared_ptr<Chunk> >::iterator it = results.begin(); it != results.end(); ++it) {
	    std::cout<< *(*it).get() << std::endl;
	}

	//

	/*	testDifferentBuffers(536870912, 4096);
	 testDifferentBuffers(536870912, 8192);
	 testDifferentBuffers(536870912, 12288);
	 testDifferentBuffers(536870912, 16384);
	 testDifferentBuffers(536870912, 20480);
	 testDifferentBuffers(536870912, 24576);
	 testDifferentBuffers(536870912, 28672);
	 testDifferentBuffers(536870912, 32768);
	 testDifferentBuffers(536870912, 36864);
	 testDifferentBuffers(536870912, 40960);
	 testDifferentBuffers(536870912, 45056);
	 testDifferentBuffers(536870912, 49152);
	 testDifferentBuffers(536870912, 53248);
	 testDifferentBuffers(536870912, 57344);
	 testDifferentBuffers(536870912, 61440);
	 testDifferentBuffers(536870912, 65536);
	 testDifferentBuffers(536870912, 69632);
	 testDifferentBuffers(536870912, 73728);
	 testDifferentBuffers(536870912, 77824);
	 testDifferentBuffers(536870912, 81920);
	 testDifferentBuffers(536870912, 86016);
	 testDifferentBuffers(536870912, 90112);
	 testDifferentBuffers(536870912, 94208);
	 testDifferentBuffers(536870912, 98304);*/

}
