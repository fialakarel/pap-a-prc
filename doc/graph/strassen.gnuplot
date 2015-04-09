set term png
set output "strassen.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "závislost času na počtu procesů -- Strassenův rekurzivní algoritmus"
set xlabel "počet vláken"
set ylabel "čas (sec)"

plot "strassen-1000" using 1:2 title '1000x1000' with line, "strassen-2000" using 1:2 title '2000x2000' with line, "strassen-3000" using 1:2 title '3000x3000' with line