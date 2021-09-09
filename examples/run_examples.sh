#!/bin/bash

for filename in *.py; do
    if [[ "$filename" == "ex10_kp.py" ]]; then
      echo Running example: "$filename"
      python3 $filename 50
    elif [[ "$filename" == "ex1_tsp.py" ]]; then
      echo Running example: "$filename"
      python3 $filename 24
    elif [[ "$filename" == "ex_nhho.py" ]]; then
      echo Running example: "$filename"
      python3 $filename 5
    else
      echo Running example: "$filename"
      python3 $filename
    fi
done

rm -rf *.png *.csv *.h5
