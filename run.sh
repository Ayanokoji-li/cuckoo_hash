#!/bin/bash

echo "---------------------"
echo "task 1 start"

for i in $(seq 10 1 24)
do
    ./task1 --repeat 5 --insert $i
done

echo "task 1 end"
echo ""
echo "---------------------"
echo "task 2 start"

./task2 --repeat 5

echo "task 2 end"
echo ""
echo "---------------------"

echo "task 3 start"

for i in $(seq 1.1 0.1 2)
do
    ./task3 --ratio $i --repeat 5
done
./task3 --ratio 1.01 --repeat 5
./task3 --ratio 1.02 --repeat 5
./task3 --ratio 1.05 --repeat 5

echo "task 3 end"
echo ""
echo "----------------------"

echo "task 4 start"
for i in $(seq 1 1 10)
do 
    ./task4 --ratio $i --repeat 5
done
echo "task 4 end"