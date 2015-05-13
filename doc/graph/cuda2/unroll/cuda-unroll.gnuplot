set term png
set output "cuda-unroll.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "CUDA -- rozbalení cyklu (i=2)"
set xlabel "velikost matice"
set ylabel "čas"

plot "cuda-unroll-100.txt" using 2:3 title '100 vláken' with line, "cuda-unroll-300.txt" using 2:3 title '300 vláken' with line, "cuda-unroll-500.txt" using 2:3 title '500 vláken' with line, "cuda-unroll-800.txt" using 2:3 title '800 vláken' with line, "cuda-unroll-1024.txt" using 2:3 title '1024 vláken' with line, "../../cuda/cuda" using 1:2 title '1024 vláken bez rozbalení' with line