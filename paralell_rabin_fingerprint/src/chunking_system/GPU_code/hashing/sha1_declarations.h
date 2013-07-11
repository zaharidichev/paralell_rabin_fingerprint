/**
 * sha1_declarations.c
 *
 *  Created on: Jul 7, 2013
 *      Author: Zahari Dichev <zaharidichev@gmail.com>
 */

#ifndef SHA1_DECLARATIONS_C_
#define SHA1_DECLARATIONS_C_
#include "../../../etc/DedupDefines.h"


extern "C" void createHashes(BYTE* data, size_t* breakpoints, BYTE* output, int numChunk);

#endif /* SHA1_DECLARATIONS_C_ */
