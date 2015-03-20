
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
//	c = s_alg(a, b);

	matrix a1 = getPart(0, 0, a);
    matrix a2 = getPart(0, 1, a);
    matrix a3 = getPart(1, 0, a);
    matrix a4 = getPart(1, 1, a);

    matrix b1 = getPart(0, 0, b);
    matrix b2 = getPart(0, 1, b);
    matrix b3 = getPart(1, 0, b);
    matrix b4 = getPart(1, 1, b);

	// x
    matrix a1_b1 = s_alg(a1, b1);
    matrix a2_b3 = s_alg(a2, b3);

	// y
	matrix a1_b2 = s_alg(a1, b2);
    matrix a2_b4 = s_alg(a2, b4);

	// z
	matrix a3_b1 = s_alg(a3, b1);
    matrix a4_b3 = s_alg(a4, b3);

	// w
	matrix a3_b2 = s_alg(a3, b2);
    matrix a4_b4 = s_alg(a4, b4);

	matrix x = addM(a1_b1, a2_b3);
	matrix y = addM(a1_b2, a2_b4);
   	matrix z = addM(a3_b1, a4_b3);
 	matrix w = addM(a3_b2, a4_b4);

	matrix c;
	c.p = Alloc(a.size);
    c.size = a.size;
    
    setPart(0, 0, &c, x);
    setPart(0, 1, &c, y);
    setPart(1, 0, &c, z);
    setPart(1, 1, &c, w);
    

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
    
    matrix c = mpWrap(a, b);
    
    
    return c.p;
    
}
