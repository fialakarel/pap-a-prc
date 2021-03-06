
typedef struct {
    int ** p;
    int size;
} matrix;


void firstAlloc() {
    
    fakeMatrix = new int**[FAKE_MATRIX_SIZE];
    
    for (int j=0; j<FAKE_MATRIX_SIZE; j++) {
        fakeMatrix[j] = new int*[size];
        for (int i=0; i<size; i++) {
            fakeMatrix[j][i] = new int[size];
            /*for (int k = 0 ; k < size ; k++) {
                fakeMatrix[j][i][k] = 0;
            }*/
        }
    }
}

int ** fakeAlloc() {
    pfm = pfm + 1;
    if (pfm > (FAKE_MATRIX_SIZE-5) ) {
        pfm = 1;
    }
    return fakeMatrix[pfm];
}

int ** Alloc(int size) {
    #ifdef FAKE_ALLOC
        return fakeAlloc();
    #else    
        return allocMatrix(size);
    #endif
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

    #ifdef FAKE_ALLOC
        // nic
    #else    
    
    for (int i=0; i<x.size; i++) {
        delete[] (x.p[i]);
    }
    delete[](x.p);
        
    #endif
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



// strassen algorithm
int ** strassen(int size, int ** A, int ** B) {
    
    matrix a;
    a.p = A;
    a.size = size;
    
    matrix b;
    b.p = B;
    b.size = size;
    
    #ifdef FAKE_ALLOC
        firstAlloc();
    #endif
    
    matrix c = s_alg(a, b);
    
    
    return c.p;
    
}