#!/bin/bash


num_projections=("10")
dataset=$2

if [ $dataset == "cifar10" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
elif [ $dataset == "svhn" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
elif [ $dataset == "fashion_mnist" ]; then
    weight_decay=5e-4
    weight_decay_theta=5e-4
fi

for proj in "${num_projections[@]}"; do
    for seed in {0..4}; do
        python ./experiments/torchvision_active_projections.py --coreset $1 --dataset $dataset --seed $seed --batch_size 3000 --budget 12000 --gamma 0.7 --num_projections $proj --init_num_labeled 1000 --num_features 32 --freq_summary 50 --weight_decay $weight_decay --weight_decay_theta $weight_decay_theta
    done
done

