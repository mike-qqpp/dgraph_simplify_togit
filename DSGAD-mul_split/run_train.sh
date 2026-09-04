#!/bin/bash

# DSGAD 训练脚本 - 用于训练模型并保存

# 设置GPU设备
GPU_ID=${1:-0}

# 数据集配置
DATASET=${2:-"dgraph"}

# 模型保存配置
MODEL_SAVE_DIR=${3:-"../models/dgraphfin"}
MODEL_NAME=${4:-"best_model"}

# 训练参数
EPOCH=${5:-5}
RUN=${6:-1}
BATCH_SIZE=${7:-5120}
PATIENCE=${8:-10}
LR=${9:-1e-3}
LR_WEIGHTS=${10:-1e-1}

# 模型配置
H_FEATS=${11:-32}
D=${12:-2}
MIX_BETA=${13:-2}
USE_REDUCED_MEMORY=${14:-true}

echo "=============================================="
echo "DSGAD 训练脚本"
echo "=============================================="
echo "GPU ID：${GPU_ID}"
echo "数据集：${DATASET}"
echo "模型保存目录：${MODEL_SAVE_DIR}"
echo "模型名称：${MODEL_NAME}"
echo "训练轮次：${EPOCH}"
echo "运行次数：${RUN}"
echo "批次大小：${BATCH_SIZE}"
echo "早停耐心值：${PATIENCE}"
echo "学习率：${LR}"
echo "权重学习率：${LR_WEIGHTS}"
echo "隐藏层维度：${H_FEATS}"
echo "阶数：${D}"
echo "混合Beta：${MIX_BETA}"
echo "使用内存优化：${USE_REDUCED_MEMORY}"
echo "=============================================="

# 检查模型配置
if [ "${USE_REDUCED_MEMORY}" = "true" ]; then
    MODEL_CONFIG="{\"h_feats\": ${H_FEATS}, \"d\": ${D}, \"mix_beta\": ${MIX_BETA}, \"use_reduced_memory\": true}"
else
    MODEL_CONFIG="{\"h_feats\": ${H_FEATS}, \"d\": ${D}, \"mix_beta\": ${MIX_BETA}, \"use_reduced_memory\": false}"
fi

echo "模型配置：${MODEL_CONFIG}"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 运行训练脚本（与原始命令格式一致）
python train_dsgad.py \
    --model DSGAD \
    --run ${RUN} \
    --dataset ${DATASET} \
    --epoch ${EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --lr_weights ${LR_WEIGHTS} \
    --patience ${PATIENCE} \
    --model_config "${MODEL_CONFIG}" \
    --model_save_dir ${MODEL_SAVE_DIR} \
    --model_name ${MODEL_NAME}

echo ""
echo "=============================================="
echo "训练完成！"
echo "模型保存路径：${MODEL_SAVE_DIR}/${MODEL_NAME}.pt"
echo "=============================================="

