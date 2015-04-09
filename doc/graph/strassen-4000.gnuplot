set term png
set output "strassen-4000.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "závislost času na počtu procesů -- Strassenův rekurzivní algoritmus"
set xlabel "počet vláken"
set ylabel "čas (sec)"

plot "strassen-4000" using 1:2 title '4000x4000' with line, 70.087705/x title 'lineární zrychlení' with line