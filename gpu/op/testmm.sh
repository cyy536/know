#!/bin/bash
for ((M=32; M<=512; M*=2)); do
    for ((K=32; K<=512; K*=2)); do
        for ((N=32; N<=512; N*=2)); do
            ./a.out "$M" "$K" "$N"
        done
    done
done