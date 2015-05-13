
typedef struct {
    int ** p;
    int size;
} matrix;


int ** Alloc(int size) {
    return allocMatrix(size);
}

__global__ void mul_gpu(int *A, int *B, int *C, int size) {

        // A, B jsou vstupni matice
        // C je vystupni matice
        // size je dim A

        int g = blockIdx.x*1024 + threadIdx.x;

        //int block = blockIdx.x;
//      int thread = threadIdx.x;

        int x = g/size;
        int y = g % size;

//      printf("Hello %dx%d\n",block, thread);

        int tmp = 0;

        for (int i = 0; i < size; i++) {
//              tmp += A[block*size + i] * B[i*size + thread];
                tmp += A[x*size + i] * B[i*size + y];
        }

        // vystup
        C[x*size + y] = tmp;

        // synchronizace pred prepnutim -- jinak dava spatny vysledek?
        __syncthreads();

}

__global__ void add_gpu(int *A, int *B, int *C, int size) {

        // A, B jsou vstupni matice
        // C je vystupni matice
        // size je dim A
        int g = blockIdx.x*1024 + threadIdx.x;
        
        // vystup
        if ( g < size ) {
            C[g] = A[g] + B[g];
        }
        
        __syncthreads();
}  

__global__ void sub_gpu(int *A, int *B, int *C, int size) {

        // A, B jsou vstupni matice
        // C je vystupni matice
        // size je dim A
        int g = blockIdx.x*1024 + threadIdx.x;
        
        // vystup
        if ( g < size ) {
            C[g] = A[g] - B[g];
        }
        
        __syncthreads();
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
    
    // mereni
    clock_t start, end;
    clock_t start_gpu, end_gpu;
    start = clock();
    int size = a.size/2;
    
    // nastaveni spusteni
    int gx = ((size*size)/1024 + 1);
    int bx = 1024;
    dim3 grid(gx, 1, 1);
    dim3 block(bx, 1, 1);
    
    // pocatecni rozdeleni
    matrix a11 = getPart(0, 0, a);
    matrix a12 = getPart(0, 1, a);
    matrix a21 = getPart(1, 0, a);
    matrix a22 = getPart(1, 1, a);
    
    matrix b11 = getPart(0, 0, b);
    matrix b12 = getPart(0, 1, b);
    matrix b21 = getPart(1, 0, b);
    matrix b22 = getPart(1, 1, b);
    
    int *cuda_a11, *cuda_a12, *cuda_a21, *cuda_a22, *cuda_b11, *cuda_b12, *cuda_b21, *cuda_b22;
    
    cudaMalloc((void**)&cuda_a11, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_a12, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_a21, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_a22, sizeof(int)*size*size);
    
    cudaMalloc((void**)&cuda_b11, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_b12, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_b21, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_b22, sizeof(int)*size*size);    
    
    // dostanu 2*4 matice do GPU pameti
    for (int i = 0; i < size; i++) {
        cudaMemcpy(&cuda_a11[i*size], a11.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(&cuda_a12[i*size], a12.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(&cuda_a21[i*size], a21.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(&cuda_a22[i*size], a22.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        
        cudaMemcpy(&cuda_b11[i*size], b11.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(&cuda_b12[i*size], b12.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(&cuda_b21[i*size], b21.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
        cudaMemcpy(&cuda_b22[i*size], b22.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
    }
    
    // toto uz nepotrebuji na CPU
    cleanM(a11);
    cleanM(a12);
    cleanM(a21);
    cleanM(a22);
    cleanM(b11);
    cleanM(b12);
    cleanM(b21);
    cleanM(b22);

    // inicializace
    int *cuda_t1, *cuda_t2, *cuda_m1, *cuda_t3, *cuda_m2,
        *cuda_t4, *cuda_m3, *cuda_t5, *cuda_m4, *cuda_t6,
        *cuda_m5, *cuda_t7, *cuda_t8, *cuda_m6, *cuda_t9, *cuda_t10, *cuda_m7;
    
    // a alokace pameti pro pomocne matice
    cudaMalloc((void**)&cuda_t1, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t2, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m1, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t3, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m2, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t4, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m3, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t5, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m4, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t6, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m5, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t7, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t8, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m6, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t9, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_t10, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_m7, sizeof(int)*size*size);
    
    start_gpu = clock();
    add_gpu<<< grid, block >>>(cuda_a11, cuda_a22, cuda_t1, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_b11, cuda_b22, cuda_t2, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_a21, cuda_a22, cuda_t3, size);
    cudaThreadSynchronize();
    sub_gpu<<< grid, block >>>(cuda_b12, cuda_b22, cuda_t4, size);
    cudaThreadSynchronize();
    sub_gpu<<< grid, block >>>(cuda_b21, cuda_b11, cuda_t5, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_a11, cuda_a12, cuda_t6, size);
    cudaThreadSynchronize();
    sub_gpu<<< grid, block >>>(cuda_a21, cuda_a11, cuda_t7, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_b11, cuda_b12, cuda_t8, size);
    cudaThreadSynchronize();
    sub_gpu<<< grid, block >>>(cuda_a12, cuda_a22, cuda_t9, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_b21, cuda_b22, cuda_t10, size);
    cudaThreadSynchronize();
    
//     matrix t1 = addM(a11, a22);
//     matrix t2 = addM(b11, b22);
//     matrix t3 = addM(a21, a22);
//     matrix t4 = subM(b12, b22);
//     matrix t5 = subM(b21, b11);
//     matrix t6 = addM(a11, a12);
//     matrix t7 = subM(a21, a11);
//     matrix t8 = addM(b11, b12);
//     matrix t9 = subM(a12, a22);
//     matrix t10 = addM(b21, b22);
    
    //cout << "po alokaci" << endl << flush;
    
//     for (int i = 0; i < size; i++) {
//         cudaMemcpy(&cuda_t1[i*size], t1.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t2[i*size], t2.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t3[i*size], t3.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_b11[i*size], b11.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_a11[i*size], a11.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t4[i*size], t4.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_a22[i*size], a22.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t5[i*size], t5.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t6[i*size], t6.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_b22[i*size], b22.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t7[i*size], t7.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t8[i*size], t8.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t9[i*size], t9.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//         cudaMemcpy(&cuda_t10[i*size], t10.p[i], sizeof(int)*size, cudaMemcpyHostToDevice);
//     }
    //cout << "po memcpy" << endl << flush;
    
//     matrix m1 = s_alg(t1, t2);
//     matrix m2 = s_alg(t3, b11);
//     matrix m3 = s_alg(a11, t4);
//     matrix m4 = s_alg(a22, t5);
//     matrix m5 = s_alg(t6, b22);
//     matrix m6 = s_alg(t7, t8);
//     matrix m7 = s_alg(t9, t10);


    
    mul_gpu<<< grid, block >>>(cuda_t1, cuda_t2, cuda_m1, size);
    cudaThreadSynchronize();
    mul_gpu<<< grid, block >>>(cuda_t3, cuda_b11, cuda_m2, size);
    cudaThreadSynchronize();
    mul_gpu<<< grid, block >>>(cuda_a11, cuda_t4, cuda_m3, size);
    cudaThreadSynchronize();
    mul_gpu<<< grid, block >>>(cuda_a22, cuda_t5, cuda_m4, size);
    cudaThreadSynchronize();
    mul_gpu<<< grid, block >>>(cuda_t6, cuda_b22, cuda_m5, size);
    cudaThreadSynchronize();
    mul_gpu<<< grid, block >>>(cuda_t7, cuda_t8, cuda_m6, size);
    cudaThreadSynchronize();
    mul_gpu<<< grid, block >>>(cuda_t9, cuda_t10, cuda_m7, size);
    cudaThreadSynchronize();
    
    
//     matrix m1, m2, m3, m4, m5, m6, m7;
//     m1.p = Alloc(size);
//     m1.size = size;
//     m2.p = Alloc(size);
//     m2.size = size;
//     m3.p = Alloc(size);
//     m3.size = size;
//     m4.p = Alloc(size);
//     m4.size = size;
//     m5.p = Alloc(size);
//     m5.size = size;
//     m6.p = Alloc(size);
//     m6.size = size;
//     m7.p = Alloc(size);
//     m7.size = size;
    
    
//     for (int i = 0; i < size; i++) {
//     //              cout << "pruchod: " << i << flush << endl;
//         cudaMemcpy(m1.p[i], &cuda_m1[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m2.p[i], &cuda_m2[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m3.p[i], &cuda_m3[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m4.p[i], &cuda_m4[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m5.p[i], &cuda_m5[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m6.p[i], &cuda_m6[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//         cudaMemcpy(m7.p[i], &cuda_m7[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
//     }
//     cudaThreadSynchronize();
    
//     printMatrix(m1.p, m1.size);
//     printMatrix(m2.p, m2.size);
//     printMatrix(m3.p, m3.size);
//     printMatrix(m4.p, m4.size);
//     printMatrix(m5.p, m5.size);
//     printMatrix(m6.p, m6.size);
//     printMatrix(m7.p, m7.size);

    // ******************************
    // pokracuji normalne

    
//     cleanM(t1);
//     cleanM(t2);
//     cleanM(t3);
//     cleanM(t4);
//     cleanM(t5);
//     cleanM(t6);
//     cleanM(t7);
//     cleanM(t8);
//     cleanM(t9);
//     cleanM(t10);
    
    cudaFree(cuda_t1);
    cudaFree(cuda_t2);
    cudaFree(cuda_t3);
    cudaFree(cuda_b11);
    cudaFree(cuda_a11);
    cudaFree(cuda_t4);
    cudaFree(cuda_a22);
    cudaFree(cuda_t5);
    cudaFree(cuda_t6);
    cudaFree(cuda_b22);
    cudaFree(cuda_t7);
    cudaFree(cuda_t8);
    cudaFree(cuda_t9);
    cudaFree(cuda_t10);

    
    matrix c;
    c.p = Alloc(a.size);
    c.size = a.size;
    
    int *cuda_rx1,*cuda_rx2, *cuda_rx3, *cuda_r2, *cuda_r3, *cuda_ry1, *cuda_ry2, *cuda_ry3;
    
    cudaMalloc((void**)&cuda_rx1, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_rx2, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_rx3, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_r2, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_r3, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_ry1, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_ry2, sizeof(int)*size*size);
    cudaMalloc((void**)&cuda_ry3, sizeof(int)*size*size);
    
    add_gpu<<< grid, block >>>(cuda_m1, cuda_m4, cuda_rx1, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_rx1, cuda_m7, cuda_rx2, size);
    cudaThreadSynchronize();
    sub_gpu<<< grid, block >>>(cuda_rx2, cuda_m5, cuda_rx3, size);
    cudaThreadSynchronize();
    
    add_gpu<<< grid, block >>>(cuda_m3, cuda_m5, cuda_r2, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_m2, cuda_m4, cuda_r3, size);
    cudaThreadSynchronize();
    
    sub_gpu<<< grid, block >>>(cuda_m1, cuda_m2, cuda_ry1, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_ry1, cuda_m3, cuda_ry2, size);
    cudaThreadSynchronize();
    add_gpu<<< grid, block >>>(cuda_ry2, cuda_m6, cuda_ry3, size);
    cudaThreadSynchronize();
    
//     matrix rx1 = addM(m1, m4);
//     matrix rx2 = addM(rx1, m7);
//     matrix rx3 = subM(rx2, m5);
//     
//     matrix r2 = addM(m3, m5);
//     matrix r3 = addM(m2, m4);
//     
//     matrix ry1 = subM(m1, m2);
//     matrix ry2 = addM(ry1, m3);
//     matrix ry3 = addM(ry2, m6);


    end_gpu = clock();
    cudaFree(cuda_m1);    
    cudaFree(cuda_m2);
    cudaFree(cuda_m3);
    cudaFree(cuda_m4);
    cudaFree(cuda_m5);
    cudaFree(cuda_m6);
    cudaFree(cuda_m7);
    
    matrix rx3, r2, r3, ry3;
    rx3.p = Alloc(size);
    rx3.size = size;
    r2.p = Alloc(size);
    r2.size = size;
    r3.p = Alloc(size);
    r3.size = size;
    ry3.p = Alloc(size);
    ry3.size = size;
    
    for (int i = 0; i < size; i++) {
        cudaMemcpy(rx3.p[i], &cuda_rx3[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
        cudaMemcpy(r2.p[i], &cuda_r2[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
        cudaMemcpy(r3.p[i], &cuda_r3[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
        cudaMemcpy(ry3.p[i], &cuda_ry3[i*size], sizeof(int)*size, cudaMemcpyDeviceToHost);
    }
    
    setPart(0, 0, &c, rx3);
    setPart(0, 1, &c, r2);
    setPart(1, 0, &c, r3);
    setPart(1, 1, &c, ry3);
    
//     cleanM(m1);
//     cleanM(m2);
//     cleanM(m3);
//     cleanM(m4);
//     cleanM(m5);
//     cleanM(m6);
//     cleanM(m7);
    
//     cleanM(rx1);
//     cleanM(rx2);
    cleanM(rx3);
    cleanM(r2);
    cleanM(r3);
//     cleanM(ry1);
//     cleanM(ry2);
    cleanM(ry3);
    
    
    // vypis mereni
    end = clock();    
    cout << "Running for " << (double)(end-start)/CLOCKS_PER_SEC << endl << flush;
    cout << "GPU running for " << (double)(end_gpu-start_gpu)/CLOCKS_PER_SEC << endl << flush;
    
 
    return c;

}
    

// strassen algorithm
int ** strassen(int size, int ** A, int ** B) {
    
    matrix a;
    a.p = A;
    a.size = size;
    
    matrix b;
    b.p = B;
    b.size = size;
    
    matrix c = s_alg(a, b);    
    
    return c.p;
    
}
