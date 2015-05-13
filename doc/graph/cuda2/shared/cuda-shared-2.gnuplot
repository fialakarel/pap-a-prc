set term png
set output "cuda-shared-2.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "CUDA -- sdílená paměť -- detail"
set xlabel "velikost matice"
set ylabel "čas"

plot "cuda-shared-8.txt" using 2:3 title 'tile=8' with line, "cuda-shared-16.txt" using 2:3 title 'tile=16' with line, "cuda-shared-24.txt" using 2:3 title 'tile=24' with line, "cuda-shared-32.txt" using 2:3 title 'tile=32' with line