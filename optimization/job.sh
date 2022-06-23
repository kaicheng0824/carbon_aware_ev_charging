#!/bin/bash
for i in 15 30 45 60 75 90 105 120 135 150 165 180
do
  echo "Running P = $i"
  python EDF_2021.py --P $i
done

