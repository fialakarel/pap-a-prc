#!/bin/bash

printf "\n\n\tCompile: \t"

g++ -std=c++11 -O3 -Wall -pedantic -Dalg_sisd_classic src/main.cpp -o matrix.out

printf "done\n"

exit $?