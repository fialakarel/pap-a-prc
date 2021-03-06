//#define GPU_TH 1024
//#define TILE 8


__global__ void mul_gpu(int *A, int *B, int *C, int size) {

	// A, B jsou vstupni matice
	// C je vystupni matice
	// size je dim A

	__shared__ int As[TILE][TILE];
	__shared__ int Bs[TILE][TILE];


	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int x = by * TILE + ty;
	int y = bx * TILE + tx;

//	printf("Hello %dx%d\n",block, thread);

	int tmp = 0;

	for (int k = 0; k < size/TILE; k++) {
		As[ty][tx] = A[x*size + k*TILE + tx];
		Bs[ty][tx] = B[y + (k*TILE + ty)*size];
	}

	__syncthreads();

	for (int i = 0; i < TILE; i++) {
		tmp  += As[ty][i] * Bs[i][tx];	
	}

	// vystup
	C[x*size + y] = tmp;

	// synchronizace pred prepnutim -- jinak dava spatny vysledek?
	__syncthreads();

}

// trivial algorithm
void trivial(int size, int ** A, int ** B, int ** C) {
 
 	clock_t start, end;
 	clock_t start_gpu, end_gpu;
 	start = clock();

 	int *cuda_A;
 	int *cuda_B;
 	int *cuda_C;

	// nastaveni spusteni

	//int gx;
	//int bx;
	/*
	if (size < 0) {
		gx = size;
		bx = size;
	} else {
		// zajistit saturaci
		bx = GPU_TH;

		gx = (((size*size)/GPU_TH) + 1)/2;
	}*/

 	dim3 grid(size/TILE, size/TILE, 1);
 	dim3 block(TILE, TILE, 1);

// 	cout << "pred alokaci" << flush << endl;

 	cudaMalloc((void**)&cuda_A, sizeof(int)*size*size);
	cudaMalloc((void**)&cuda_B, sizeof(int)*size*size);
	cudaMalloc((void**)&cuda_C, sizeof(int)*size*size);

//	cout << "pred kopirovanim" << flush << endl;

	for (int i = 0; i < size; i++) {
//		cout << "pruchod: " << i << flush << endl;
		cudaMemcpy(&cuda_A[i*size], A[i], sizeof(int)*size, cudaMemcpyHostToDevice);
		cudaMemcpy(&cuda_B[i*size], B[i], sizeof(int)*size, cudaMemcpyHostToDevice);
	}
//	cudaMemcpy(cuda_A, A, sizeof(int)*size*size, cudaMemcpyHostToDevice);
//	cudaMemcpy(cuda_B, B, sizeof(int)*size*size, cudaMemcpyHostToDevice);

//	cout << "pred spustenim" << flush << endl;
 
 	start_gpu = clock();

	mul_gpu<<< grid, block >>>(cuda_A, cuda_B, cuda_C, size);

	end_gpu = clock();

//	cout << "pred synchronizaci kernelu" << flush << endl;
 
	cudaThreadSynchronize();

//	cout << "pred dolovanim vysledku" << flush << endl;
 
//	cudaMemcpy(C, cuda_C, sizeof(int)*size*size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < size; i++) {
//		cout << "pruchod: " << i << flush << endl;
		cudaMemcpy(C[i], &cuda_C[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
	}


//	cout << "pred uvolneni pameti" << flush << endl;
 
	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);

//	cout << "pred ukoncenim" << flush << endl;

	end = clock();

	cout << "Running for " << (double)(end-start)/CLOCKS_PER_SEC << endl << flush;
	cout << "GPU running for " << (double)(end_gpu-start_gpu)/CLOCKS_PER_SEC << endl << flush;

}
