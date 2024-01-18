#! /usr/bin/bash

# Compile task1
nvcc -O3 -Xcompiler -O3 -o task1 task1.cu

# Compile task2
nvcc -O3 -Xcompiler -O3 -o task2 task2.cu

# Compile task3
nvcc -O3 -Xcompiler -O3 -o task3 task3.cu

# Compile task4
nvcc -O3 -Xcompiler -O3 -o task4 task4.cu