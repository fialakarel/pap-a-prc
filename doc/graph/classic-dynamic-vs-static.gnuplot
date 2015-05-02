set term png
set output "classic-dynamic-vs-static.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
#unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Graf zrychlení -- Classic -- porovnání dynamic vs static"
set xlabel "počet vláken"
set ylabel "zrychlení"

plot "classic-4000-acc" using 1:2 title 'Classic 4000x4000 dynamic' with line,\
    "classic-4000-static-acc" using 1:2 title 'Classic 4000x4000 static' with line