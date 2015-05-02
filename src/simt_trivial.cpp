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
