set term png
set output "classic-3000.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "závislost času na počtu procesů -- klasický multiplikační algoritmus"
set xlabel "počet vláken"
set ylabel "čas (sec)"

plot "classic-3000" using 1:2 title '3000x3000' with line, 264.885942/x title 'lineární zrychlení' with line