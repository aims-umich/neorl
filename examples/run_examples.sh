#!/bin/bash

unset DISPLAY
numexamples=$(find *.py -type f | wc -l)
index=1
for filename in *.py; do
    echo Running example $index/$numexamples: "$filename"
    if [[ "$filename" == "ex10_kp.py" ]]; then
      python3 $filename 50 > /dev/null
    elif [[ "$filename" == "ex1_tsp.py" ]]; then
      python3 $filename 24 > /dev/null
    elif [[ "$filename" == "ex_nhho.py" ]]; then
      python3 $filename 5 > /dev/null
    elif [[ "$filename" == "ex_nhho_discrete.py" ]]; then
      python3 $filename 2 > /dev/null
    else
      python3 $filename > /dev/null
    fi
	index=$((index+1))
done

rm -rf *.png *.csv *.h5
