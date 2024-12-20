#!/bin/bash

datasets=("caltech101" "dtd" "eurosat" "fgvc" "food101" \
          "imagenet" "oxford_flowers" "oxford_pets" "stanford_cars" "sun397" "ucf101")

exp=1
step_size=$(echo "scale=1; 1 / 10 ^ $exp" | bc)
alpha_list=$(seq 0 $step_size 1)
shots_list=(16 8 4 2 1)


for dataset in "${datasets[@]}"; do
  for shots in "${shots_list[@]}"; do
    for alpha in $alpha_list; do
      for beta in $(seq 0 $step_size $(echo "scale=1; 1 - $alpha" | bc)); do
        beta=$(printf "%.1f" $beta)
        gamma=$(echo "scale=1; 1 - $alpha - $beta" | bc)
        gamma=$(printf "%.1f" $gamma)

        sum=$(echo "$alpha + $beta + $gamma" | bc)
        sum=$(printf "%.1f" $sum)

        if (( $(echo "$alpha >= 0" | bc -l) )) && (( $(echo "$beta >= 0" | bc -l) )) && (( $(echo "$gamma >= 0" | bc -l) )) && (( $(echo "$sum == 1.0" | bc -l) )); then
          echo "Running with dataset=$dataset, alpha=$alpha, beta=$beta, gamma=$gamma, shots=$shots"
          CUDA_VISIBLE_DEVICES=0 python train.py --config configs/$dataset.yml --dataset $dataset --alpha $alpha --beta $beta --shots $shots
        fi
      done
    done
  done
done
