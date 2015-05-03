
__global__ void mul_gpu(int *A, int *B, int *C, int size) {

	// A, B jsou vstupni matice
	// C je vystupni matice
	// size je dim A

	int g = blockIdx.x*1024 + threadIdx.x;

	//int block = blockIdx.x;
//	int thread = threadIdx.x;

	int x = g/size;
	int y = g % size;

//	printf("Hello %dx%d\n",block, thread);

	int tmp = 0;

	for (int i = 0; i < size; i++) {
//		tmp += A[block*size + i] * B[i*size + thread];
		tmp += A[x*size + i] * B[i*size + y];
	}

	// vystup
	C[x*size + y] = tmp;

	// synchronizace pred prepnutim -- jinak dava spatny vysledek?
	__syncthreads();

}

// trivial algorithm
void trivial(int size, int ** A, int ** B, int ** C) {
 
 	int *cuda_A;
 	int *cuda_B;
 	int *cuda_C;

	// nastaveni spusteni

	int gx;
	int bx;
	if (size < 1025) {
		gx = size;
		bx = size;
	} else {
		// zajistit saturaci
		bx = 1024;

		gx = ((size*size)/1024) + 1;
	}

 	dim3 grid(gx, 1, 1);
 	dim3 block(bx, 1, 1);

 	cout << "pred alokaci" << flush << endl;

 	cudaMalloc((void**)&cuda_A, sizeof(int)*size*size);
	cudaMalloc((void**)&cuda_B, sizeof(int)*size*size);
	cudaMalloc((void**)&cuda_C, sizeof(int)*size*size);

	cout << "pred kopirovanim" << flush << endl;

	for (int i = 0; i < size; i++) {
//		cout << "pruchod: " << i << flush << endl;
		cudaMemcpy(&cuda_A[i*size], A[i], sizeof(int)*size, cudaMemcpyHostToDevice);
		cudaMemcpy(&cuda_B[i*size], B[i], sizeof(int)*size, cudaMemcpyHostToDevice);
	}
//	cudaMemcpy(cuda_A, A, sizeof(int)*size*size, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_B, B, sizeof(int)*size*size, cudaMemcpyHostToDevice);

	cout << "pred spustenim" << flush << endl;
 
	mul_gpu<<< grid, block >>>(cuda_A, cuda_B, cuda_C, size);

	cout << "pred synchronizaci kernelu" << flush << endl;
 
	cudaThreadSynchronize();

	cout << "pred dolovanim vysledku" << flush << endl;
 
//	cudaMemcpy(C, cuda_C, sizeof(int)*size*size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++) {
//		cout << "pruchod: " << i << flush << endl;
		cudaMemcpy(C[i], &cuda_C[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
	}


	cout << "pred uvolneni pameti" << flush << endl;
 
	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);

	cout << "pred ukoncenim" << flush << endl;

}
