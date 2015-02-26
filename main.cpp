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

int ** loadFromFile(int m, int n, string filePath) {
    string line;
    ifstream file(filePath);
    int linenumber = 0;
    
    // Check if its correctly open
    if (file.is_open()) {
        // Alloc triangle
        
        int ** matrix = new int*[m];
        
        for (int i=0; i<m; i++) {
            matrix[i] = new int[n];
        }
        
        
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


int main (int argc, char **argv) {

    // Check bad number of parameters
    if (argc != 7) {
        printf("\n\n\tusage:\t./a.out m n matA n p matB\n\n\n");
        return 1;
    }
    
    // Store arguments from command line
    // Format is ./a.out m n matA n p matB
    string matA_file = argv[3];
    string matB_file = argv[6];
    int matA_m = stoi(argv[1]);
    int matA_n = stoi(argv[2]);
    int matB_n = stoi(argv[4]);
    int matB_p = stoi(argv[5]);
    
    // debug print
    cout << matA_m << "x" << matA_n << " -- " << matA_file << endl;
    cout << matB_n << "x" << matB_p << " -- " << matB_file << endl;
    
    
    // Init default values
    init();
    
    matA = loadFromFile(matA_m, matA_n, matA_file);
    matB = loadFromFile(matB_n, matB_p, matB_file);
    
    cout << "matA: " << endl;
    for(int i = 0; i < matA_m; i++) {
        for(int j = 0; j < matA_m; j++) {
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
        
    // Run main procces
    //mainProccesLoop();
    
    
    // Clean up data
    //cleanUp();
    
    return 0;
}
