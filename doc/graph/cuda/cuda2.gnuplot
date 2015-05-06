set term png
set output "cuda2.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "CUDA -- klasický multiplikační algoritmus"
set xlabel "velikosti matice"
set ylabel "(N^3)/čas"

plot "cuda2" using 1:2 title 'implementace' with line