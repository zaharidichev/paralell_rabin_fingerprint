/**
 * trigger.cpp
 *
 *  Created on: Jul 2, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#include "stdio.h"
#include "GPUChunker.h"
#include "../GPU_code/BitFieldArray.h"

typedef unsigned char BYTE;
BYTE* allocateData(int size) {

	BYTE* data = (BYTE*) malloc(sizeof(BYTE) * size);
	srand(2);
	for (int var = 0; var < size; ++var) {
		data[var] = (BYTE) rand() % 256;
	}

	return data;

}

int main() {

	int sizeOfData = 536870912;
	int minSize = 32768;
	int maxSize = 131072;

	//const char* dataToChunk = (const char*)allocateData(sizeOfData);

	  std::ifstream infile ("/home/zahari/Desktop/data.txt",std::ofstream::binary);
	  unsigned char* buffer = (unsigned char*)malloc(33363);

	  //infile.seekg(67076482);

	    infile.seekg (535857225);

	  infile.read((char*)buffer,33363);
	  BYTE* digest = (BYTE*)malloc(20);

	  SHA1(buffer,33363,digest);

	  for (int var = 0; var < 20; ++var) {
	        printf("%02x", digest[var]);

	}
	 printf("\n");

	  //outfile.write (dataToChunk,sizeOfData);


	infile.seekg (0, infile.beg);
	GPUChunker chunker = GPUChunker(512, 0xbfe6b8a5bf378d83,minSize,maxSize);

	 vector<shared_ptr<Chunk> > chunks =  chunker.chunkDataFromFile(infile,sizeOfData);


//
	 for(std::vector<shared_ptr<Chunk> >::iterator it = chunks.begin(); it != chunks.end(); ++it) {
		 std::cout << *((*it).get()) << std::endl;
	 }




	//free(dataToChunk);

}
