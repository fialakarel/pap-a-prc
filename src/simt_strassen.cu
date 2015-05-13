
typedef struct {
    int ** p;
    int size;
} matrix;


int ** Alloc(int size) {
    return allocMatrix(size);
}


matrix multM(matrix a, matrix b) {
    matrix c;
    c.p = Alloc(a.size);
    c.size = a.size;
    for (int i = 0; i < a.size; i++) {
        for (int j = 0; j < a.size; j++) {
            int result = 0;
            for (int k = 0; k < a.size; k++) {
                result += a.p[i][k] * b.p[k][j];
            }
            c.p[i][j] = result;
        }
    }
    return c;
}

matrix subM(matrix a, matrix b) {
    matrix c;
    c.p = Alloc(a.size);
    c.size = a.size;
    for (int i = 0 ; i < a.size ; i++) {
        for (int j = 0 ; j < a.size ; j++) {
            c.p[i][j] = a.p[i][j] - b.p[i][j];
        }
    }
    return c;
}

matrix addM(matrix a, matrix b) {
    matrix c;
    c.p = Alloc(a.size);
    c.size = a.size;
    for (int i = 0 ; i < a.size ; i++) {
        for (int j = 0 ; j < a.size ; j++) {
            c.p[i][j] = a.p[i][j] + b.p[i][j];
        }
    }
    return c;
}

matrix getPart(int f1, int f2, matrix x) {
    matrix c;
    c.p = Alloc(x.size/2);
    c.size = x.size/2;
    int xstart = f1 * c.size ;
    int ystart = f2 * c.size ;
    
    for (int i = 0 ; i < c.size ; i++) {
        for (int j = 0 ; j < c.size ; j++) {
            c.p[i][j] = x.p[i + xstart][j + ystart];
        }
    }
    return c;
}

void setPart(int f1, int f2, matrix *target, matrix source) {
    int xstart = f1 * source.size ;
    int ystart = f2 * source.size ;
    
    for (int i = 0 ; i < source.size ; i++) {
        for (int j = 0 ; j < source.size ; j++) {
            target->p[i + xstart][j + ystart] = source.p[i][j];
        }
    }
}


void cleanM(matrix x) {

    for (int i=0; i<x.size; i++) {
        delete[] (x.p[i]);
    }
    delete[](x.p);
        
}


matrix s_alg(matrix a, matrix b) {
    
    if ( a.size <= STRASSEN_THRESHOLD ) {
        return multM(a, b);
    }
    
    //printMatrix(a.p, a.size);
    
    matrix a11 = getPart(0, 0, a);
    matrix a12 = getPart(0, 1, a);
    matrix a21 = getPart(1, 0, a);
    matrix a22 = getPart(1, 1, a);
    
    matrix b11 = getPart(0, 0, b);
    matrix b12 = getPart(0, 1, b);
    matrix b21 = getPart(1, 0, b);
    matrix b22 = getPart(1, 1, b);
    
    matrix t1 = addM(a11, a22);
    matrix t2 = addM(b11, b22);
    matrix t3 = addM(a21, a22);
    matrix t4 = subM(b12, b22);
    matrix t5 = subM(b21, b11);
    matrix t6 = addM(a11, a12);
    matrix t7 = subM(a21, a11);
    matrix t8 = addM(b11, b12);
    matrix t9 = subM(a12, a22);
    matrix t10 = addM(b21, b22);

    matrix m1 = s_alg(t1, t2);
    matrix m2 = s_alg(t3, b11);
    matrix m3 = s_alg(a11, t4);
    matrix m4 = s_alg(a22, t5);
    matrix m5 = s_alg(t6, b22);
    matrix m6 = s_alg(t7, t8);
    matrix m7 = s_alg(t9, t10);

    cleanM(a11);
    cleanM(a12);
    cleanM(a21);
    cleanM(a22);
    cleanM(b11);
    cleanM(b12);
    cleanM(b21);
    cleanM(b22);
    
    cleanM(t1);
    cleanM(t2);
    cleanM(t3);
    cleanM(t4);
    cleanM(t5);
    cleanM(t6);
    cleanM(t7);
    cleanM(t8);
    cleanM(t9);
    cleanM(t10);

    
    matrix c;
    c.p = Alloc(a.size);
    c.size = a.size;
    
    matrix rx1 = addM(m1, m4);
    matrix rx2 = addM(rx1, m7);
    matrix rx3 = subM(rx2, m5);
    
    matrix r2 = addM(m3, m5);
    matrix r3 = addM(m2, m4);
    
    matrix ry1 = subM(m1, m2);
    matrix ry2 = addM(ry1, m3);
    matrix ry3 = addM(ry2, m6);
    
    
    setPart(0, 0, &c, rx3);
    setPart(0, 1, &c, r2);
    setPart(1, 0, &c, r3);
    setPart(1, 1, &c, ry3);
    
    cleanM(m1);
    cleanM(m2);
    cleanM(m3);
    cleanM(m4);
    cleanM(m5);
    cleanM(m6);
    cleanM(m7);
    
    cleanM(rx1);
    cleanM(rx2);
    cleanM(rx3);
    cleanM(r2);
    cleanM(r3);
    cleanM(ry1);
    cleanM(ry2);
    cleanM(ry3);
    
 
    return c;

}

matrix mpWrap(matrix a, matrix b) {

	// multiprocess here
	int depth = 2;
	int parts = 16;
        
    int conf[64][2] =
    {
    	{0,0},
    	{1,2},
    	{4,8},
    	{5,10},
    	{0,1},
    	{1,3},
    	{4,9},
    	{5,11},
    	{0,4},
    	{1,6},
    	{4,12},
    	{5,14},
    	{0,5},
    	{1,7},
    	{4,13},
    	{5,15},
    	{2,0},
    	{3,2},
    	{6,8},
    	{7,10},
    	{2,1},
    	{3,3},
    	{6,9},
    	{7,11},
    	{2,4},
    	{3,6},
    	{6,12},
    	{7,14},
    	{2,5},
    	{3,7},
    	{6,13},
    	{7,15},
    	{8,0},
    	{9,2},
    	{12,8},
    	{13,10},
    	{8,1},
    	{9,3},
    	{12,9},
    	{13,11},
    	{8,4},
    	{9,6},
    	{12,12},
    	{13,14},
    	{8,5},
    	{9,7},
    	{12,13},
    	{13,15},
    	{10,0},
    	{11,2},
    	{14,8},
    	{15,10},
    	{10,1},
    	{11,3},
    	{14,9},
    	{15,11},
    	{10,4},
    	{11,6},
    	{14,12},
    	{15,14},
    	{10,5},
    	{11,7},
    	{14,13},
    	{15,15}
    };

	// pripravit nove, mensi matice
	matrix ** e = new matrix*[depth+1];
	matrix ** f = new matrix*[depth+1];

	// alokovat nové matice
	int ti = 0;
	for (int i = 1; i < parts+1; i = i*4) {
		e[ti] = new matrix[i];
		f[ti] = new matrix[i];
		ti++;
	}

	// pripravit vstupni pole 
	e[0][0] = a;
	f[0][0] = b;

	int pos = 1;
	int pointer = 0;

	// rozpulit matice na PARTS do hloubky DEPTH
	for (int i = 1; i < depth+1; i++) {

		pointer = 0;

		for (int j = 0; j < pos; j++) {

			e[i][pointer] = getPart(0, 0, e[i-1][j]);
			e[i][pointer+1] = getPart(0, 1, e[i-1][j]);
			e[i][pointer+2] = getPart(1, 0, e[i-1][j]);
			e[i][pointer+3] = getPart(1, 1, e[i-1][j]);
	
			f[i][pointer] = getPart(0, 0, f[i-1][j]);
			f[i][pointer+1] = getPart(0, 1, f[i-1][j]);
			f[i][pointer+2] = getPart(1, 0, f[i-1][j]);
			f[i][pointer+3] = getPart(1, 1, f[i-1][j]);

			pointer = pointer + 4;
		}

		pos = pos * 4;
    }

#ifdef DEBUG_PRINT
	cout << "a.size: " << a.size << endl;
	for (int j = 0 ; j < parts; j++) {
		cout << "A[" << depth << "][" << j << "]" << endl;
		printMatrix(e[depth][j].p, e[depth][j].size);
		cout << "B[" << depth << "][" << j << "]" << endl;
		printMatrix(f[depth][j].p, f[depth][j].size);
	
	}
	cout << "*******" << endl;
#endif



    // nastavit pocet vláken
    omp_set_num_threads(THREADS);


	// inicializace vysledných matic
    matrix * r = new matrix[parts*4];
    int i;
	
	// paralelni zpracovani matic
	#pragma omp parallel for shared(r, i) schedule(dynamic,1)
	for (i = 0; i < parts*4; i++) {

		r[i] = s_alg(e[depth][conf[i][0]], f[depth][conf[i][1]]);
//		cout << "i: " << i << " ei: " << ei << " fi: " << fi << endl << flush;
	}

	
#ifdef DEBUG_PRINT
	for (int j = 0; j < parts*4; j++) {
		printMatrix(r[j].p, r[j].size);
	}
	cout << "*******" << endl;
#endif


	// prvni redukce 64 > 32
	#pragma omp parallel for shared(r, i) schedule(dynamic,1)
	for (i = 0; i < parts*4 ; i = i + 2) {
		r[i] = addM(r[i], r[i+1]);
	}

	// druha redukce 32 -> 16
	#pragma omp parallel for shared(r, i) schedule(dynamic,1)
	for (i = 0; i < parts*4; i = i + 4) {
		r[i] = addM(r[i], r[i+2]);
	}

//	matrix * cx = new matrix[parts];
#ifdef DEBUG_PRINT
	for (int j = 0; j < parts * 4; j = j + 4) {
		printMatrix(r[j].p, r[j].size);
	}
#endif
	matrix t[4];

	// skladani 16 -> 4
	#pragma omp parallel for shared(t, r, i) schedule(dynamic,1)
	for (i = 0; i < 4; i++) {
		t[i].p = Alloc(a.size/2);
		t[i].size = a.size/2;

		int tmp = 0;
		
		if (i > 2) {
			tmp = 16;
		}

	    setPart(0, 0, &t[i], r[tmp + i*8 +  0]);
	    setPart(0, 1, &t[i], r[tmp + i*8 +  4]);
	    setPart(1, 0, &t[i], r[tmp + i*8 + 16]);
	    setPart(1, 1, &t[i], r[tmp + i*8 + 20]);
	}

#ifdef DEBUG_PRINT
	cout << "skladani" << endl;
	for (int j = 0; j < 4; j++) {
		printMatrix(t[j].p, t[j].size);
	}
	cout << "*******" << endl;
#endif



	matrix c;
	c.p = Alloc(a.size);
    c.size = a.size;
    
    setPart(0, 0, &c, t[0]);
    setPart(0, 1, &c, t[1]);
    setPart(1, 0, &c, t[2]);
    setPart(1, 1, &c, t[3]);
    

	return c;
}
        
// *****************************************************
        
__global__ void mul_gpu(int *A, int *B, int *C, int size) {

        // A, B jsou vstupni matice
        // C je vystupni matice
        // size je dim A



        int bx = blockIdx.x;
        int by = blockIdx.y;

        int tx = threadIdx.x;
        int ty = threadIdx.y;


        int x = by * TILE + ty;
        int y = bx * TILE + tx;

//      printf("Hello %dx%d\n",block, thread);

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
        
        

// *****************************************************
// strassen algorithm
int ** strassen(int size, int ** A, int ** B) {
    
    matrix a;
    a.p = A;
    a.size = size;
    
    matrix b;
    b.p = B;
    b.size = size;
    
    matrix c;
    
    clock_t start, end;
    clock_t start_gpu, end_gpu;
    start = clock();
    
    int *cuda_A0;
    int *cuda_A1;
    int *cuda_A2;
    int *cuda_A3;
    
    int *cuda_B0;
    int *cuda_B1;
    int *cuda_B2;
    int *cuda_B3;
    
    int *cuda_C0;
    int *cuda_C1;
    int *cuda_C2;
    int *cuda_C3;
    
    matrix a0, a1, a2, a3, b0, b1, b2, b3, c0, c1, c2, c3;
    a0 = getPart(0, 0, a);
    a1 = getPart(0, 1, a);
    a2 = getPart(1, 0, a);
    a3 = getPart(1, 1, a);
    
    b0 = getPart(0, 0, b);
    b1 = getPart(0, 1, b);
    b2 = getPart(1, 0, b);
    b3 = getPart(1, 1, b);
    
    // nastaveni spusteni
    
    //int gx;
    //int bx;
    
    if (size < 0) {
        gx = size;
        bx = size;
    } else {
        // zajistit saturaci
        bx = 1024;
    
        gx = (((size*size)/1024) + 1)/4;
    }
    
    dim3 grid(gx, 1, 1);
    dim3 block(bx, 1, 1);
    
    //      cout << "pred alokaci" << flush << endl;
    
    cudaMalloc((void**)&cuda_A0, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_A1, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_A2, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_A3, sizeof(int)*((size*size)/4));
    
    cudaMalloc((void**)&cuda_B0, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_B1, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_B2, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_B3, sizeof(int)*((size*size)/4));
    
    cudaMalloc((void**)&cuda_C0, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_C1, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_C2, sizeof(int)*((size*size)/4));
    cudaMalloc((void**)&cuda_C3, sizeof(int)*((size*size)/4));
    
    //      cout << "pred kopirovanim" << flush << endl;

    
    
    for (int i = 0; i < size/2; i++) {
        //              cout << "pruchod: " << i << flush << endl;
            cudaMemcpy(&cuda_A0[i*(size/2)], a0[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
            cudaMemcpy(&cuda_A1[i*(size/2)], a1[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
            cudaMemcpy(&cuda_A2[i*(size/2)], a2[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
            cudaMemcpy(&cuda_A3[i*(size/2)], a3[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);

            
            cudaMemcpy(&cuda_B0[i*(size/2)], b0[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
            cudaMemcpy(&cuda_B1[i*(size/2)], b1[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
            cudaMemcpy(&cuda_B2[i*(size/2)], b2[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
            cudaMemcpy(&cuda_B3[i*(size/2)], b3[i], sizeof(int)*(size/2), cudaMemcpyHostToDevice);
        }
    //      cudaMemcpy(cuda_A, A, sizeof(int)*size*size, cudaMemcpyHostToDevice);
    //      cudaMemcpy(cuda_B, B, sizeof(int)*size*size, cudaMemcpyHostToDevice);
    
    //      cout << "pred spustenim" << flush << endl;
    
    start_gpu = clock();
    
    // gpu vypocet
    mul_gpu<<< grid, block >>>(cuda_A0, cuda_B0, cuda_C0, size/2);
    mul_gpu<<< grid, block >>>(cuda_A1, cuda_B1, cuda_C1, size/2);
    mul_gpu<<< grid, block >>>(cuda_A2, cuda_B2, cuda_C2, size/2);
    mul_gpu<<< grid, block >>>(cuda_A3, cuda_B3, cuda_C3, size/2);

    
    end_gpu = clock();
    
    //      cout << "pred synchronizaci kernelu" << flush << endl;
    
    cudaThreadSynchronize();
    
    //      cout << "pred dolovanim vysledku" << flush << endl;
    
    //      cudaMemcpy(C, cuda_C, sizeof(int)*size*size, cudaMemcpyDeviceToHost);
    
    c0.p = Alloc(a.size/2);
    c0.size = a.size/2;
    c1.p = Alloc(a.size/2);
    c1.size = a.size/2;
    c2.p = Alloc(a.size/2);
    c2.size = a.size/2;
    c3.p = Alloc(a.size/2);
    c3.size = a.size/2;
    
    for (int i = 0; i < size/2; i++) {
        //              cout << "pruchod: " << i << flush << endl;
            cudaMemcpy(c0[i], &cuda_C0[i*(size/2)], sizeof(int)*(size/2), cudaMemcpyDeviceToHost);
            cudaMemcpy(c1[i], &cuda_C1[i*(size/2)], sizeof(int)*(size/2), cudaMemcpyDeviceToHost);
            cudaMemcpy(c2[i], &cuda_C2[i*(size/2)], sizeof(int)*(size/2), cudaMemcpyDeviceToHost);
            cudaMemcpy(c3[i], &cuda_C3[i*(size/2)], sizeof(int)*(size/2), cudaMemcpyDeviceToHost);
        }
    
    c.p = Alloc(a.size);
    c.size = a.size;
    
    setPart(0, 0, &c, c0);
    setPart(0, 1, &c, c1);
    setPart(1, 0, &c, c2);
    setPart(1, 1, &c, c3);
    
    //      cout << "pred uvolneni pameti" << flush << endl;
    
    cudaFree(cuda_A0);
    cudaFree(cuda_A1);
    cudaFree(cuda_A2);
    cudaFree(cuda_A3);
    cudaFree(cuda_B0);
    cudaFree(cuda_B1);
    cudaFree(cuda_B2);
    cudaFree(cuda_B3);
    cudaFree(cuda_C0);
    cudaFree(cuda_C1);
    cudaFree(cuda_C2);
    cudaFree(cuda_C3);
    
    //      cout << "pred ukoncenim" << flush << endl;
    
    end = clock();
    
    cout << "Running for " << (double)(end-start)/CLOCKS_PER_SEC << endl << flush;
    cout << "GPU running for " << (double)(end_gpu-start_gpu)/CLOCKS_PER_SEC << endl << flush;
    
    
    
    
    
    return c.p;
    
}
