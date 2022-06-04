#!/bin/bash

numexamples=$(find *.py -type f | wc -l)
index=1
for filename in *.py; do
    if [[ "$filename" == "ex10_kp.py" ]]; then
      echo Running example $index/$numexamples: "$filename"
      python3 $filename 50 > /dev/null
    elif [[ "$filename" == "ex1_tsp.py" ]]; then
      echo Running example $index/$numexamples: "$filename"
      python3 $filename 24 > /dev/null
    elif [[ "$filename" == "ex_nhho.py" ]]; then
      echo Running example $index/$numexamples: "$filename"
      python3 $filename 5 > /dev/null
    else
      echo Running example $index/$numexamples: "$filename"
      python3 $filename > /dev/null
    fi
	index=$((index+1))
done

rm -rf *.png *.csv *.h5
