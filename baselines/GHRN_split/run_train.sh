#!/bin/bash

# 训练脚本 - 用于训练模型并保存

# 设置GPU设备
GPU_ID=${1:-0}

# 数据集配置
DATASET=${2:-"dgraphfin"}
DATA_PATH=${3:-"../data"}
MODEL_SAVE_DIR=${4:-"../models/GHRN/dgraphfin"}
MODEL_NAME=${5:-"best_model"}

# 训练参数
EPOCH=${6:-1}
RUN=${7:-1}
TRAIN_BATCH_SIZE=${8:-5120}
EVAL_BATCH_SIZE=${9:-512}

# 截断参数（可选）
TRAIN_TRUNCATE=${10:-0}
VAL_TRUNCATE=${11:-0}
TEST_TRUNCATE=${12:-0}
TRUNCATE_MODE=${13:-"random"}

# 模型参数
HID_DIM=${14:-64}
ORDER=${15:-2}
HOMO=${16:-1}

echo "=============================================="
echo "BWGNN 训练脚本"
echo "=============================================="
echo "GPU ID: ${GPU_ID}"
echo "数据集: ${DATASET}"
echo "数据路径: ${DATA_PATH}"
echo "模型保存目录: ${MODEL_SAVE_DIR}"
echo "模型名称: ${MODEL_NAME}"
echo "训练轮次: ${EPOCH}"
echo "运行次数: ${RUN}"
echo "训练批次大小: ${TRAIN_BATCH_SIZE}"
echo "评估批次大小: ${EVAL_BATCH_SIZE}"
echo "训练集截断: ${TRAIN_TRUNCATE}"
echo "验证集截断: ${VAL_TRUNCATE}"
echo "测试集截断: ${TEST_TRUNCATE}"
echo "截断模式: ${TRUNCATE_MODE}"
echo "隐藏层维度: ${HID_DIM}"
echo "阶数: ${ORDER}"
echo "同构/异构: ${HOMO}"
echo "=============================================="

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 运行训练脚本
python train_split.py \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_save_dir ${MODEL_SAVE_DIR} \
    --model_name ${MODEL_NAME} \
    --epoch ${EPOCH} \
    --run ${RUN} \
    --train_batch_size ${TRAIN_BATCH_SIZE} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --train_truncate ${TRAIN_TRUNCATE} \
    --val_truncate ${VAL_TRUNCATE} \
    --test_truncate ${TEST_TRUNCATE} \
    --truncate_mode ${TRUNCATE_MODE} \
    --hid_dim ${HID_DIM} \
    --order ${ORDER} \
    --homo ${HOMO} \
    --gpu ${GPU_ID}

echo ""
echo "=============================================="
echo "训练完成!"
echo "模型保存路径: ${MODEL_SAVE_DIR}/${MODEL_NAME}.pt"
echo "=============================================="

