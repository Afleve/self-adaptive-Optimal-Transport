datasets=("sicap_mil" "skincancer" "lc_lung" "nct" "WSSS4LUAD")

# 运行所有数据集并收集结果
for dataset in "${datasets[@]}"; do
    echo "正在处理数据集：$dataset"
    
    # 运行TransCLIP模型
    CUDA_VISIBLE_DEVICES=1 python main.py --dataset "$dataset"  \
                                    --root "js散度"
done