set term png
set output "classic-vs-strassen.png"
set grid  xtics ytics
set   autoscale                        # scale axes automatically
#unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "závislost času na počtu procesů -- porovnání algoritmů"
set xlabel "počet výpočetních jednotek"
set ylabel "čas (sec)"

plot "strassen-1000" using 1:2 title 'Strassen 1000x1000' with line,\
    "strassen-2000" using 1:2 title 'Strassen 2000x2000' with line,\
    "strassen-3000" using 1:2 title 'Strassen 3000x3000' with line,\
    "strassen-4000" using 1:2 title 'Strassen 4000x4000' with line,\
    "classic-1000" using 1:2 title 'Classic 1000x1000' with line,\
    "classic-2000" using 1:2 title 'Classic 2000x2000' with line,\
    "classic-3000" using 1:2 title 'Classic 3000x3000' with line,\
    "classic-4000" using 1:2 title 'Classic 4000x4000' with line