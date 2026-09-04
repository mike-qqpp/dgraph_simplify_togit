#!/bin/bash

# DSGAD 推理基准测试脚本 - 包含CPU运行时长和内存测量
# 使用方法: ./run_infer_benchmark.sh [GPU_ID] [DATASET] [MODEL_PATH] [BATCH_SIZE] [SAVE_PREDICTIONS] [PREDICTIONS_PATH]

# 记录总开始时间（使用纳秒级精度）
TOTAL_START_TIME=$(date +%s.%N)

# 设置默认参数
GPU_ID=${1:-0}
DATASET=${2:-"dgraph"}
MODEL_PATH=${3:-"../models/dgraphfin/best_model.pt"}
BATCH_SIZE=${4:-512}
SAVE_PREDICTIONS=${5:-"false"}
PREDICTIONS_PATH=${6:-"./predictions.npz"}

echo "=============================================="
echo "DSGAD 推理基准测试脚本"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo "GPU ID: ${GPU_ID}"
echo "数据集: ${DATASET}"
echo "模型路径: ${MODEL_PATH}"
echo "批次大小: ${BATCH_SIZE}"
echo "保存预测: ${SAVE_PREDICTIONS}"
echo "预测保存路径: ${PREDICTIONS_PATH}"
echo "=============================================="

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误：模型文件不存在：${MODEL_PATH}"
    exit 1
fi

# 检查数据集文件是否存在
DATA_SPLIT_PATH="../data_split/${DATASET}_test.npz"
if [ ! -f "${DATA_SPLIT_PATH}" ]; then
    echo "警告：数据集文件 ${DATA_SPLIT_PATH} 不存在，但将继续执行..."
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 初始化变量
INFER_TIME=""
INFER_MEMORY=""

# 构建参数
PYTHON_ARGS="--dataset ${DATASET} \
    --model_path ${MODEL_PATH} \
    --batch_size ${BATCH_SIZE} \
    --device cpu"

# 添加保存预测参数
if [ "${SAVE_PREDICTIONS}" = "true" ]; then
    PYTHON_ARGS="${PYTHON_ARGS} --save_predictions --predictions_path ${PREDICTIONS_PATH}"
fi

echo ""
echo ">>> 开始DSGAD模型推理..."
INFER_START_TIME=$(date +%s.%N)

# 使用/usr/bin/time测量内存使用
if command -v /usr/bin/time >/dev/null 2>&1; then
    # 创建临时文件存储时间和内存信息
    TEMP_TIME_FILE=$(mktemp)
    
    # 执行推理脚本，使用/usr/bin/time捕获内存
    TEMP_OUTPUT=$(mktemp)
    /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$TEMP_TIME_FILE" \
        python infer_dsgad_cpu.py ${PYTHON_ARGS} > "$TEMP_OUTPUT" 2>&1
    EXIT_CODE=$?
    
    # 显示命令输出
    cat "$TEMP_OUTPUT"
    rm -f "$TEMP_OUTPUT"
    
    INFER_END_TIME=$(date +%s.%N)
    
    # 提取内存使用
    MEMORY_KB=$(grep "MEMORY_KB:" "$TEMP_TIME_FILE" | cut -d: -f2)
    if [ -n "$MEMORY_KB" ] && echo "$MEMORY_KB" | grep -q "^[0-9]\+$"; then
        INFER_MEMORY=$(echo "$MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        INFER_MEMORY="N/A"
    fi
    
    rm -f "$TEMP_TIME_FILE"
else
    # 如果没有/usr/bin/time，只测量时间
    python infer_dsgad_cpu.py ${PYTHON_ARGS}
    EXIT_CODE=$?
    INFER_END_TIME=$(date +%s.%N)
    INFER_MEMORY="N/A"
fi

# 计算推理时间
INFER_TIME=$(echo "$INFER_END_TIME $INFER_START_TIME" | awk '{printf "%.4f", $1 - $2}')

if [ $EXIT_CODE -ne 0 ]; then
    echo "错误：推理失败，退出代码: $EXIT_CODE"
    exit $EXIT_CODE
fi

# 总结束时间
TOTAL_END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$TOTAL_END_TIME $TOTAL_START_TIME" | awk '{printf "%.4f", $1 - $2}')

echo ""
echo "=============================================="
echo "推理基准测试结果汇总"
echo "=============================================="
printf "%-25s %-15s %-20s\n" "步骤" "耗时(秒)" "峰值内存(MB)"
echo "------------------------------------------------"
printf "%-25s %-15s %-20s\n" "DSGAD推理" "${INFER_TIME}" "${INFER_MEMORY}"
echo "------------------------------------------------"
printf "%-25s %-15s\n" "总执行时间" "${TOTAL_TIME}"
echo "=============================================="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# 检查/usr/bin/time是否可用
if [ "${INFER_MEMORY}" = "N/A" ]; then
    echo ""
    echo "注意: /usr/bin/time 不可用，内存测量不可用"
    echo "如需内存测量，请安装: apt-get update && apt-get install -y time"
fi

# 保存测试结果到文件
RESULT_LOG="inference_benchmark_results.log"
echo "$(date '+%Y-%m-%d %H:%M:%S'),${DATASET},${INFER_TIME},${INFER_MEMORY},${BATCH_SIZE}" >> ${RESULT_LOG}
echo "结果已追加到: ${RESULT_LOG}"
