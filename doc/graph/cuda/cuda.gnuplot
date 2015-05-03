set term png
set output "cuda1.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "závislost času na velikosti matice -- CUDA -- klasický multiplikační algoritmus"
set xlabel "velikosti matice"
set ylabel "čas (sec)"

plot "cuda" using 1:2 title 'implementace' with line, 0.000206814*x title 'přímka' with line