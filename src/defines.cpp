
// ukazatele na matice
int ** matA = NULL;
int ** matB = NULL;
int ** matC = NULL;

// soubory vstupnich matic
string matA_file;
string matB_file;

// matrix dimension
int size;
int prev_size;

int *** fakeMatrix = NULL;
int pfm = 0;
int FAKE_MATRIX_SIZE = 100;

int STRASSEN_THRESHOLD = 500;

//#define FAKE_ALLOC 1
//

//#define DEBUG_PRINT 1

#ifndef THREADS
	#define THREADS 2
#endif
