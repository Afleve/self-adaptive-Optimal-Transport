# 定义要运行的所有数据集
datasets=( "imagenet" "sun397" "fgvc" "eurosat"  "stanford_cars"  "food101"  "oxford_pets" "oxford_flowers" "caltech101" "dtd" "ucf101")

# 运行所有数据集并收集结果
for dataset in "${datasets[@]}"; do
    echo "正在处理数据集：$dataset"
    # 运行TransCLIP模型，使用零样本设置
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset "$dataset" --method OptimalTrans --shots 0 --backbone vit_b16 --root "res_5_iters" --device "cuda"

done