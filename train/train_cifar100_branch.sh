#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

data="cifar100"
adapter="lora"
# epochs=100

for i in $(seq 1 50); do
  save_name="${adapter}_focal_${i}"
  echo "Running: data=${data}, adapter=${adapter}, epochs=${epochs}, save_path=${save_name}"

  python train/train.py \
    --data $data \
    --adapter $adapter \
    --netsize s \
    --save_path $save_name \
    --gpu 7
done
