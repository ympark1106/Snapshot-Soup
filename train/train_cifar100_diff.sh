#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

data="cifar100"
adapter="lora"
# epochs=100

# Define learning rates for different phases
learning_rates=("1e-3" "1e-4" "1e-5")

# Loop over the learning rates, 20 iterations for each
for learning_rate in "${learning_rates[@]}"; do
  for i in $(seq 1 20); do 
    save_name="${adapter}_focal_lr_${learning_rate}_${i}"
    echo "Running: data=${data}, adapter=${adapter}, learning_rate=${learning_rate}, save_path=${save_name}"

    python train/train_diff.py \
      --data $data \
      --adapter $adapter \
      --netsize s \
      --save_path $save_name \
      --gpu 3 \
      --learning_rate $learning_rate  
  done
done
