#!/bin/bash
case $1 in
    "normal")
        python3 evaluate.py --dataset-folder datasets/benchmark_datasets
        ;;
esac