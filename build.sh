#!/bin/bash

printf "\n\n\tCompile: \t"

t="alg_simd_classic"

if [ "$1" == "classic" ]
then
    t="alg_simd_classic"
fi

if [ "$1" == "strassen" ]
then
    t="alg_simd_strassen"
fi


g++ -fopenmp -std=c++11 -O3 -Wall -pedantic -D$t src/main.cpp -o matrix.out
#g++ -std=c++11 -ggdb -Wall -pedantic -D$t src/main.cpp -o matrix.out

printf "done\n"

exit $?
