#!/bin/bash

for size in 3 5 10 50 100
do
    printf "\n\n\t***** ***** ***** ***** ***** *****\n\n"
    time ./matrix.out $size $size sample/test-${size}x${size} $size $size sample/test-${size}x${size}

done

exit 0