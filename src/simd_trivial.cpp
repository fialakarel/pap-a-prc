// trivial algorithm
void trivial(int size, int ** A, int ** B, int ** C) {
   
    int i;
	omp_set_num_threads(THREADS);

    #pragma omp parallel for shared(A, B, C) private(i) schedule(dynamic,10)
    for (i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int result = 0;
            
            for (int k = 0; k < size; k++) {
                    result += A[i][k] * B[k][j];
                    //cout << result << " " << matA[i][k] << " " <<  matB[k][j] << endl << flush;
            }
            //#pragma omp critical 
            C[i][j] = result;
        }
    }
}
