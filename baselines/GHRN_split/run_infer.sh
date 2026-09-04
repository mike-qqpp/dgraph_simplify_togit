#!/bin/bash

# 推理脚本 - 用于加载模型并对测试集进行推理

# 设置GPU设备
GPU_ID=${1:-0}

# 数据集配置
DATASET=${2:-"dgraphfin"}
DATA_PATH=${3:-"../data"}

# 模型配置
MODEL_PATH=${4:-"../models/GHRN/dgraphfin/best_model.pt"}

# 评估参数
EVAL_BATCH_SIZE=${5:-512}
HID_DIM=${6:-64}
ORDER=${7:-2}
HOMO=${8:-1}

# 输出配置
SAVE_PREDICTIONS=${9:-"false"}
PREDICTIONS_PATH=${10:-"./predictions.npz"}

echo "=============================================="
echo "BWGNN 推理脚本"
echo "=============================================="
echo "GPU ID: ${GPU_ID}"
echo "数据集: ${DATASET}"
echo "数据路径: ${DATA_PATH}"
echo "模型路径: ${MODEL_PATH}"
echo "评估批次大小: ${EVAL_BATCH_SIZE}"
echo "隐藏层维度: ${HID_DIM}"
echo "阶数: ${ORDER}"
echo "同构/异构: ${HOMO}"
echo "保存预测: ${SAVE_PREDICTIONS}"
echo "预测保存路径: ${PREDICTIONS_PATH}"
echo "=============================================="

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误: 模型文件不存在: ${MODEL_PATH}"
    exit 1
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 构建参数
PYTHON_ARGS="--dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --model_path ${MODEL_PATH} \
    --eval_batch_size ${EVAL_BATCH_SIZE} \
    --hid_dim ${HID_DIM} \
    --order ${ORDER} \
    --homo ${HOMO} \
    --gpu ${GPU_ID}"

# 添加保存预测参数
if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --save_predictions --predictions_path ${PREDICTIONS_PATH}"
fi

# 运行推理脚本
python infer_split.py ${PYTHON_ARGS}

echo ""
echo "=============================================="
echo "推理完成!"
echo "=============================================="

