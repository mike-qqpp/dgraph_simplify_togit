#!/bin/bash

# DSGAD 推理脚本 - 用于加载模型并对测试集进行推理

# 设置GPU设备
GPU_ID=${1:-0}

# 数据集配置
DATASET=${2:-"dgraph"}

# 模型配置
MODEL_PATH=${3:-"../models/dgraphfin/best_model.pt"}

# 评估参数
BATCH_SIZE=${4:-512}

# 输出配置
SAVE_PREDICTIONS=${5:-"false"}
PREDICTIONS_PATH=${6:-"./predictions.npz"}

echo "=============================================="
echo "DSGAD 推理脚本"
echo "=============================================="
echo "GPU ID：${GPU_ID}"
echo "数据集：${DATASET}"
echo "模型路径：${MODEL_PATH}"
echo "批次大小：${BATCH_SIZE}"
echo "保存预测：${SAVE_PREDICTIONS}"
echo "预测保存路径：${PREDICTIONS_PATH}"
echo "=============================================="

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误：模型文件不存在：${MODEL_PATH}"
    exit 1
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 构建参数
PYTHON_ARGS="--dataset ${DATASET} \
    --model_path ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --device cuda"

# 添加保存预测参数
if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --save_predictions --predictions_path ${PREDICTIONS_PATH}"
fi

# 运行推理脚本
python infer_dsgad.py ${PYTHON_ARGS}

echo ""
echo "=============================================="
echo "推理完成！"
echo "=============================================="

