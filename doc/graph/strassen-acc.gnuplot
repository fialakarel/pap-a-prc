set term png
set output "strassen-acc.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Graf zrychlení -- Strassenův rekurzivní algoritmus"
set xlabel "počet vláken"
set ylabel "čas (sec)"

plot "strassen-acc" using 1:2 title '4000x4000' with line, 1*x title 'lineární zrychlení'