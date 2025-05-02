#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

datasets=("tinyimagenet")
epochs=(30 50)
adapters=("rein" "lora" "adaptformer")

for data in "${datasets[@]}"; do
  for epoch in "${epochs[@]}"; do
    for adapter in "${adapters[@]}"; do

    #   save_name="${data}_${adapter}_${epoch}ep_post"
      echo "Running: data=${data}, adapter=${adapter}, epochs=${epoch}"

      python train/train_more.py \
        --data $data \
        --adapter $adapter \
        --netsize s \
        --save_path post \
        --num_epoch $epoch \
        --gpu 6

    done
  done
done
