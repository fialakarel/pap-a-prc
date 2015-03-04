#include <iostream>
#include <cstring> // memcpy
#include <cstdio>
#include <sstream>
#include <fstream>

#define DEBUG 1

using namespace std;

#include "defines.cpp"



int shiftSize(int size) {
    int shifts = 0;
    int new_size = size;
    while (new_size > 1) {
        //cout << "." << flush;
        new_size>>=1;
        shifts++;
    }
    while (new_size < size) {
        //cout << "/" << flush;
        new_size<<=1;
    }
    return new_size;
}


int ** allocMatrix(int old_size) {
    
    int size = old_size;
    
    // only for Strassen
    #ifdef alg_sisd_strassen
    size = shiftSize(old_size);
    #endif
    
    int ** matrix = new int*[size];
    
    for (int i=0; i<size; i++) {
        matrix[i] = new int[size];
    }
    
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            matrix[i][j] = 0;
        }
    }
    
    return matrix;
}



#ifdef alg_sisd_classic
    #include "sisd_trivial.cpp"
#endif

#ifdef alg_sisd_strassen
    #include "sisd_strassen.cpp"
#endif

void mainProccesLoop() {
    #ifdef alg_sisd_classic
        trivial(size, matA, matB, matC);
    #endif

    #ifdef alg_sisd_strassen
        matC = strassen(size, matA, matB);
    #endif
}

void printMatrix(int ** matrix, int size) {
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            cout << matC[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void debugMatrix() {
    cout << endl << "matA: " << endl;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            cout << matA[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl << "matB: " << endl;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            cout << matB[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl << "matC: " << endl;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            cout << matC[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int ** loadFromFile(int size, string filePath) {
    string line;
    ifstream file(filePath);
    int linenumber = 0;
    
    // Check if its correctly open
    if (file.is_open()) {
        // Alloc triangle
        
        int ** matrix = allocMatrix(size);
        
        
        int number;
        
        // Load all lines
        while (getline(file, line)) {
            // Procces line
            stringstream ss;
            ss << line;
            
            // Load numbers to triangle
            for (int i=0; i<size; i++) {
                // Get number from line
                ss >> number;
                // Save number to triangle
                matrix[linenumber][i] = number;
            }
            linenumber++;
        }
        // Close file
        file.close();
        
        return matrix;
    } else {
        return NULL;
    }
}

void init() {
    // initialization
}

void cleanUp() {
    
    // free matA
    for (int i=0; i<size; i++) {
        delete[] (matA[i]);
    }
    delete[](matA);
    
    // free matB
    for (int i=0; i<size; i++) {
        delete[] (matB[i]);
    }
    delete[](matB);
    
    // free matc
    for (int i=0; i<size; i++) {
        delete[] (matC[i]);
    }
    delete[] (matC);
}


int main (int argc, char **argv) {

    // Check bad number of parameters
    if (argc != 4) {
        printf("\n\n\tusage:\t./a.out size matA matB\n\n\n");
        return 1;
    }
    
    // Store arguments from command line
    // Format is ./a.out size matA matB
    size = stoi(argv[1]);
    matA_file = argv[2];
    matB_file = argv[3];
    
    // Init default values
    init();
    
    // load input matrix
    matA = loadFromFile(size, matA_file);
    matB = loadFromFile(size, matB_file);
    matC = allocMatrix(size);
    
    prev_size = size;
    
    // only for Strassen
    #ifdef alg_sisd_strassen
        size = shiftSize(size);
    #endif
    
    // Run main procces
    mainProccesLoop();

    // debugMatrix
    /*if (matA_m < 11) {
        debugMatrix();    
    }
    */
    
    printMatrix(matC, prev_size);
    
    // Clean up data
    cleanUp();
    
    return 0;
}

