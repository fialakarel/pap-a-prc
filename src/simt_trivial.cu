
__global__ void mul_gpu(int *A, int *B, int *C, int size) {

	// A, B jsou vstupni matice
	// C je vystupni matice
	// size je dim A
	
	int block = blockIdx.x;
	int thread = threadIdx.x;

	int tmp = 0;

	for (int i = 0; i < size; i++) {
		tmp += A[block*size + i] + B[i*size + thread];
	}

	// vystup
	C[block*size + thread];

	// synchronizace pred prepnutim -- jinak dava spatny vysledek?
	__syncthreads();

}

// trivial algorithm
void trivial(int size, int ** A, int ** B, int ** C) {
   

    int i;

    for (i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int result = 0;
            
            for (int k = 0; k < size; k++) {
                    result += A[i][k] * B[k][j];
            }
            C[i][j] = result;
        }
    }
}
