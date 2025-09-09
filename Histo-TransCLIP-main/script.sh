#!/bin/bash

# 指定使用哪块 GPU，比如使用 GPU 0
export CUDA_VISIBLE_DEVICES=1

dataset=("sicap_mil" "skincancer" "lc_lung" "nct" "WSSS4LUAD" )

for dataset in "${dataset[@]}"; do
    # 运行 Python 文件
    # echo "Running $dataset"
    python main.py --dataset "$dataset" --selected_model "CLIP,CONCH,PLIP,MUSK"
done