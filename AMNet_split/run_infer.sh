#!/bin/bash
# 推理脚本（带模式选择）

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置参数
DATA_PATH="../data/amazon.npz"
MODEL_PATH="../models/AMNet/amazon/model_exp1_best.pth"  # 训练好的模型文件
OUTPUT_DIR="./inference_results"  # 输出目录
BATCH_SIZE=512
THRESHOLD=0.5
SEED=42

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 执行推理
python infer_split.py \
    --data_path $DATA_PATH \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --threshold $THRESHOLD \
    --seed $SEED

echo "推理完成！结果保存在: $OUTPUT_DIR"
