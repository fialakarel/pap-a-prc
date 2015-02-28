// trivial algorithm
void trivial() {
    for (int i = 0; i < matA_m; i++) {
        for (int j = 0; j < matB_p; j++) {
            int result = 0;
            
            for (int k = 0; k < matA_n; k++) {
                    result += matA[i][k] * matB[k][j];
                    //cout << result << " " << matA[i][k] << " " <<  matB[k][j] << endl << flush;
            }
            
            matC[i][j] = result;
        }
    }
}
