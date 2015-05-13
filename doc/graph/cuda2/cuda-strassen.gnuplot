set term png
set output "cuda-strassen.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Strassenův algoritmus -- CUDA"
set xlabel "velikost matice"
set ylabel "čas"

plot "cuda-strassen.log" using 1:2 title 'Strassen -- CUDA' with line, "../cuda/cuda" using 1:2 title 'Klasický multiplikativní alg. -- CUDA' with line