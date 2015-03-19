// trivial algorithm
void trivial(int size, int ** A, int ** B, int ** C) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int result = 0;
            
            for (int k = 0; k < size; k++) {
                    result += A[i][k] * B[k][j];
                    //cout << result << " " << matA[i][k] << " " <<  matB[k][j] << endl << flush;
            }
            
            C[i][j] = result;
        }
    }
}
