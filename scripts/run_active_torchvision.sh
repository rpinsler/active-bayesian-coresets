#!/bin/bash


batch_sizes=("3000")
init_num_labeled=1000
budget=12000
dataset=$3

if [ $dataset == "cifar10" ]; then
    weight_decay=5e-4
    batch_sizes=("5000")
    init_num_labeled=5000
    budget=20000
elif [ $dataset == "svhn" ]; then
    weight_decay=5e-4
elif [ $dataset == "fashion_mnist" ]; then
    weight_decay=5e-4
fi

for batch_size in "${batch_sizes[@]}"; do
    for seed in {0..4}; do
        python ./experiments/torchvision_active.py --acq $1 --coreset $2 --dataset $dataset --batch_size $batch_size --seed $seed --budget $budget --init_num_labeled $init_num_labeled --num_features 32 --freq_summary 50 --weight_decay $weight_decay
    done
done
