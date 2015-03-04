#!/bin/bash

printf "\n\n\tCompile: \t"

t="alg_sisd_classic"

if [ "$1" == "classic" ]
then
    t="alg_sisd_classic"
fi

if [ "$1" == "strassen" ]
then
    t="alg_sisd_strassen"
fi


g++ -std=c++11 -O3 -Wall -pedantic -D$t src/main.cpp -o matrix.out

printf "done\n"

exit $?