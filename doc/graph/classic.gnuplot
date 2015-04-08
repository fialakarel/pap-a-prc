set term png
set output "classic.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "závislost času na počtu procesů -- klasický multiplikační algoritmus"
set xlabel "počet výpočetních jednotek"
set ylabel "čas (sec)"

plot "classic-1000" using 1:2 title '1000x1000' with line, "classic-2000" using 1:2 title '2000x2000' with line, "classic-3000" using 1:2 title '3000x3000' with line, "classic-4000" using 1:2 title '4000x4000' with line