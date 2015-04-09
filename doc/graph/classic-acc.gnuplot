set term png
set output "classic-acc.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Graf zrychlení -- klasický multiplikační algoritmus"
set xlabel "počet vláken"
set ylabel "Zrychlení"

plot "classic-4000-acc" using 1:2 title '4000x4000' with line, 1*x title 'lineární zrychlení'

#, "classic-2000" using 2:1 title '2000x2000' with line, "classic-3000" using 2:1 title '3000x3000' with line, "classic-4000" using 2:1 title '4000x4000' with line