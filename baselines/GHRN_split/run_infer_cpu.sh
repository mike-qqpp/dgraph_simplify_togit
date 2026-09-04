#!/bin/bash

# 推理脚本 - 直接调用Python脚本并测量CPU运行时间和内存
# 使用方法: ./run_infer_cpu.sh [数据集] [数据路径] [模型路径] [评估批次大小] [隐藏层维度] [阶数] [同构/异构]
DATASET=${1:-"dgraphfin"}
DATA_PATH=${2:-"../data"}
MODEL_PATH=${3:-"../models/GHRN/dgraphfin/best_model.pt"}
EVAL_BATCH_SIZE=${4:-512}
HID_DIM=${5:-64}
ORDER=${6:-2}
HOMO=${7:-1}

# 记录总开始时间（使用纳秒级精度）
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_PATH}"
echo "========================================"

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误: 模型文件不存在: ${MODEL_PATH}"
    exit 1
fi

# 初始化变量
INFERENCE_TIME=""
INFERENCE_MEMORY=""

# ========== Inference ==========
echo ""
echo ">>> Running Inference..."
START_TIME=$(date +%s.%N)
python infer_split_cpu.py \
    --dataset "${DATASET}" \
    --data_path "${DATA_PATH}" \
    --model_path "${MODEL_PATH}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --hid_dim "${HID_DIM}" \
    --order "${ORDER}" \
    --homo "${HOMO}"
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
INFERENCE_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Inference failed with exit code $EXIT_CODE"
    exit 1
fi

# 测量内存（完全按照run_fe_test.sh的方式）
if command -v /usr/bin/time >/dev/null 2>&1; then
    INFERENCE_MEMORY_KB=$(/usr/bin/time -f "%M" python infer_split_cpu.py \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --model_path "${MODEL_PATH}" \
        --eval_batch_size "${EVAL_BATCH_SIZE}" \
        --hid_dim "${HID_DIM}" \
        --order "${ORDER}" \
        --homo "${HOMO}" 2>&1 >/dev/null | tail -1)
    if echo "$INFERENCE_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        INFERENCE_MEMORY=$(echo "$INFERENCE_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        INFERENCE_MEMORY="N/A"
    fi
else
    INFERENCE_MEMORY="N/A"
fi

echo "Inference completed in $INFERENCE_TIME seconds"
if [ "$INFERENCE_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${INFERENCE_MEMORY} MB"
fi

# ========== Summary ==========
TOTAL_END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$TOTAL_END_TIME $TOTAL_START_TIME" | awk '{printf "%.4f", $1 - $2}')

echo ""
echo "========================================"
echo "Execution Summary"
echo "========================================"
printf "%-35s %-12s %-20s\n" "Step" "Time(s)" "Peak Memory(MB)"
echo "------------------------------------------------------------"
printf "%-35s %-12s %-20s\n" "Inference" "$INFERENCE_TIME" "$INFERENCE_MEMORY"
echo "------------------------------------------------------------"
echo "Total Time: $TOTAL_TIME seconds"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 检查 /usr/bin/time 是否可用
if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo ""
    echo "注意: /usr/bin/time 不可用，内存测量不可用"
    echo "如需内存测量，请安装: apt-get update && apt-get install -y time"
fi

