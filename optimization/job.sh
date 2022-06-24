#!/bin/bash
for i in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180
do
  echo "Running P = $i"
  python carbon_aware.py --P $i --factor 1
  python TOU_2021.py --P $i 
done

