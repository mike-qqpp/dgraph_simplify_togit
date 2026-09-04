#!/bin/bash
# 训练验证脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 设置参数
DATA_PATH="../data/dgraphfin.npz"
SAVE_DIR="../models/AMNet/dgraphfin"  # 模型保存目录
HIDDEN_CHANNELS=32
M=3
K=2
LR=1e-3
LR_F=1e-1
WEIGHT_DECAY=5e-4
EPOCHS=10 #50
BETA=0.5
PATIENCE=50
EXP_NUM=1
EVAL_INTERVAL=10
BATCH_SIZE=512
SEED=42

# 创建保存目录
mkdir -p $SAVE_DIR

# 执行训练
python train_split.py \
    --data_path $DATA_PATH \
    --save_dir $SAVE_DIR \
    --hidden_channels $HIDDEN_CHANNELS \
    --M $M \
    --K $K \
    --lr $LR \
    --lr_f $LR_F \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --beta $BETA \
    --patience $PATIENCE \
    --exp_num $EXP_NUM \
    --eval_interval $EVAL_INTERVAL \
    --batch_size $BATCH_SIZE \
    --use_class_weight \
    --seed $SEED

echo "训练完成！模型保存在: $SAVE_DIR"
