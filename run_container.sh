#!/bin/bash
case $2 in 
    0) echo "Running on device 0: $1"
        docker run --rm --name $1 --gpus '"device=0,3"' \
        --shm-size=10g \
        -v /home/delia/implementations/PointNetVlad-Pytorch:/PointNetVlad-Pytorch \
        -v /media/SSD_DATA1/delia/pnv/benchmark_datasets:/PointNetVlad-Pytorch/datasets/benchmark_datasets \
        -it \
        delia/pointnetvlad  
        ;;
    1) echo "Running on device 1: $1"
        docker run --rm --name $1 --gpus '"device=1"' \
        --shm-size=10g \
        -v /home/delia/implementations/PointNetVlad-Pytorch:/PointNetVlad-Pytorch \
        -v /media/SSD_DATA1/delia/pnv/benchmark_datasets:/PointNetVlad-Pytorch/datasets/benchmark_datasets \
        -it \
        delia/pointnetvlad  
        ;;
    2) echo "Running on device 2: $1"
        docker run --rm --name $1 --gpus '"device=2"' \
        --shm-size=10g \
        -v /home/delia/implementations/PointNetVlad-Pytorch:/PointNetVlad-Pytorch \
        -v /media/SSD_DATA1/delia/pnv/benchmark_datasets:/PointNetVlad-Pytorch/datasets/benchmark_datasets \
        -it \
        delia/pointnetvlad  
        ;;
    3) echo "Running on device 3: $1"
        docker run --rm --name $1 --gpus '"device=3"' \
        --shm-size=10g \
        -v /home/delia/implementations/PointNetVlad-Pytorch:/PointNetVlad-Pytorch \
        -v /media/SSD_DATA1/delia/pnv/benchmark_datasets:/PointNetVlad-Pytorch/datasets/benchmark_datasets \
        -it \
        delia/pointnetvlad
        ;;
esac