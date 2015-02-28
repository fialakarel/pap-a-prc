#include <iostream>
#include <cstring> // memcpy
#include <cstdio>
#include <sstream>
#include <fstream>

#ifdef SMART_DEBUG
    #define DEBUG 1
#endif

using namespace std;

int ** matA = NULL;
int ** matB = NULL;
int ** matC = NULL;

string matA_file;
string matB_file;
int matA_m;
int matA_n;
int matB_n;
int matB_p;

/*
typedef struct {
    int x;
    int y;
} Coord;
*/

#include "trivial.cpp"

void mainProccesLoop() {
    trivial();
}


void debugMatrix() {
    cout << endl << "matA: " << endl;
    for(int i = 0; i < matA_m; i++) {
        for(int j = 0; j < matA_n; j++) {
            cout << matA[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl << "matB: " << endl;
    for(int i = 0; i < matB_n; i++) {
        for(int j = 0; j < matB_p; j++) {
            cout << matB[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl << "matC: " << endl;
    for(int i = 0; i < matA_m; i++) {
        for(int j = 0; j < matB_p; j++) {
            cout << matC[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int ** allocMatrix(int m, int n) {
    int ** matrix = new int*[m];
    
    for (int i=0; i<m; i++) {
        matrix[i] = new int[n];
    }
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            matrix[i][j] = 0;
        }
    }
    
    return matrix;
}

int ** loadFromFile(int m, int n, string filePath) {
    string line;
    ifstream file(filePath);
    int linenumber = 0;
    
    // Check if its correctly open
    if (file.is_open()) {
        // Alloc triangle
        
        int ** matrix = allocMatrix(m, n);
        
        
        int number;
        
        // Load all lines
        while (getline(file, line)) {
            // Procces line
            stringstream ss;
            ss << line;
            
            // Load numbers to triangle
            for (int i=0; i<n; i++) {
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
    for (int i=0; i<matA_m; i++) {
        delete(matA[i]);
    }
    delete(matA);
    
    // free matB
    for (int i=0; i<matB_n; i++) {
        delete(matB[i]);
    }
    delete(matB);
}


int main (int argc, char **argv) {

    // Check bad number of parameters
    if (argc != 7) {
        printf("\n\n\tusage:\t./a.out m n matA n p matB\n\n\n");
        return 1;
    }
    
    // Store arguments from command line
    // Format is ./a.out m n matA n p matB
    matA_file = argv[3];
    matB_file = argv[6];
    matA_m = stoi(argv[1]);
    matA_n = stoi(argv[2]);
    matB_n = stoi(argv[4]);
    matB_p = stoi(argv[5]);
    
    // debug print
    cout << endl << matA_m << "x" << matA_n << " -- " << matA_file << endl;
    cout << matB_n << "x" << matB_p << " -- " << matB_file << endl;
    
    
    // Init default values
    init();
    
    // load input matrix
    matA = loadFromFile(matA_m, matA_n, matA_file);
    matB = loadFromFile(matB_n, matB_p, matB_file);
    matC = allocMatrix(matA_m, matB_p);
    
    // Run main procces
    mainProccesLoop();

    // debugMatrix
    //debugMatrix();    
    
    // Clean up data
    cleanUp();
    
    return 0;
}
