
typedef struct {
    int ** p;
    int size;
} matrix;

matrix multM(matrix a, matrix b) {
    matrix c;
    c.p = allocMatrix(a.size);
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
    c.p = allocMatrix(a.size);
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
    c.p = allocMatrix(a.size);
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
    c.p = allocMatrix(x.size/2);
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


matrix s_alg(matrix a, matrix b) {
    
    if ( a.size == 1 ) {
        return multM(a, b);
    }
    
    matrix a11 = getPart(0, 0, a);
    matrix a12 = getPart(0, 1, a);
    matrix a21 = getPart(1, 0, a);
    matrix a22 = getPart(1, 1, a);
    
    matrix b11 = getPart(0, 0, b);
    matrix b12 = getPart(0, 1, b);
    matrix b21 = getPart(1, 0, b);
    matrix b22 = getPart(1, 1, b);
    
    matrix m1 = s_alg(addM(a11, a22), addM(b11, b22));
    matrix m2 = s_alg(addM(a21, a22), b11);
    matrix m3 = s_alg(a11, subM(b12, b22));
    matrix m4 = s_alg(a22, subM(b21, b11));
    matrix m5 = s_alg(addM(a11, a12), b22);
    matrix m6 = s_alg(subM(a21, a11), addM(b11, b12));
    matrix m7 = s_alg(subM(a12, a22), addM(b21, b22));
    
    matrix c;
    c.p = allocMatrix(a.size);
    c.size = a.size;
    
    setPart(0, 0, &c, subM(addM(addM(m1, m4), m7), m5));
    setPart(0, 1, &c, addM(m3, m5));
    setPart(1, 0, &c, addM(m2, m4));
    setPart(1, 1, &c, addM(addM(subM(m1, m2), m3), m6));
    
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