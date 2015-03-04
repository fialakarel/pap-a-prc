#!/bin/bash

fail=0
tmp=$(mktemp)

printf "\n\n\t=== ===== ===== Simple tests ===== ===== ===\n\tTest size\t\tmatrix\t\tdiff\n"
# do some simple tests
for size in 3 5 10
do
    printf "\t$size\t\t\t"
    ./matrix.out $size sample/jednotkova-${size}x${size} sample/test-${size}x${size} >$tmp 
    f=$?
    if [ $f -eq 0 ]
    then
        printf "[OK]\t\t"
    else
        printf "[FAIL]\t\t"
    fi
    fail=$((fail+$f))
    diff $tmp sample/result_jednotkova-${size}x${size}_test-${size}x${size} &>logs/log_jednotkova-${size}x${size}_test-${size}x${size}
    f=$?
    if [ $f -eq 0 ]
    then
        printf "[OK]"
    else
        printf "[FAIL]"
    fi
    fail=$((fail+$f))
    printf "\n"    
done

printf "\n\n\t=== ===== ===== Regular tests ===== ===== ===\n\tTest size\t\tmatrix\t\tdiff\n"
# do some regular tests
for size in 3 5 10 50 100
do
    printf "\t$size\t\t\t"
    ./matrix.out $size sample/test-${size}x${size} sample/test-${size}x${size} >$tmp
    f=$?
    if [ $f -eq 0 ]
    then
        printf "[OK]\t\t"
    else
        printf "[FAIL]\t\t"
    fi
    fail=$((fail+$f))
    diff $tmp sample/result_test-${size}x${size} &>logs/log_test-${size}x${size}
    f=$?
    if [ $f -eq 0 ]
    then
        printf "[OK]"
    else
        printf "[FAIL]"
    fi
    fail=$((fail+$f))
    printf "\n"    
done



if [ "$1" = "stress" ]
then
    if [ $fail -eq 0 ]
    then
        printf "\n\n\t=== ===== ===== Stress tests ===== ===== ===\n\tTest size\t\tmatrix\n"
        # do some stress tests
        for size in 500 1000 2000
        do
            printf "\t$size\t\t\t"
            time ./matrix.out $size sample/stress-${size}x${size} $size sample/stress-${size}x${size} &>/dev/null
            f=$?
            if [ $f -eq 0 ]
            then
                printf "[OK]"
            else
                printf "[FAIL]"
            fi
            fail=$((fail+$f))
            printf "\n\n\n"    
        done
    else
        printf "\n\n\t!!! Failed on simple tests !!!\n"
    fi
fi

rm $tmp
printf "\n\n"

exit 0



# samples:
# ========
# jednotkova-3x3
# jednotkova-5x5
# jednotkova-10x10
# 
# test-3x3
# test-5x5
# test-10x10
# test-50x50
# test-100x100
# 
# stress-1000x1000
# stress-2000x2000
# stress-500x500