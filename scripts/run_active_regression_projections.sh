#!/bin/bash

# defaults
batch_sizes=("10")
init_num_labeled=20
initial_lr=1e-2
num_units=30
budget=100
weight_decay=1.
b0=1.
seeds=40


dataset=$1
if [ $dataset == "yacht" ]; then
    :
elif [ $dataset == "energy" ]; then
    :
elif [ $dataset == "boston" ]; then
    :
elif [ $dataset == "power" ]; then
    initial_lr=1e-3
    b0=3.
    weight_decay=3.
elif [ $dataset == "year" ]; then
    seeds=5
    batch_sizes=("1000")
    init_num_labeled=200
    num_units=100
    budget=10000
    b0=3.
    weight_decay=3.
fi

for batch_size in "${batch_sizes[@]}"; do
    for ((seed=0; seed<$seeds; seed++)); do
        python ./experiments/linear_regression_active_projections.py --dataset $dataset --num_projections $2 --seed $seed --batch_size $batch_size --init_num_labeled $init_num_labeled --initial_lr $initial_lr --budget $budget --num_units $num_units --b0 $b0 --weight_decay $weight_decay # --use_gpu
    done
done
