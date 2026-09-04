#!/bin/bash

# 推理脚本（CPU版本，带运行时长和内存测量）
# 使用方法: ./run_infer_cpu.sh

# 设置参数
DATA_PATH="../data/dgraphfin.npz"
MODEL_PATH="../models/AMNet/dgraphfin/model_exp1_best.pth"
OUTPUT_DIR="./inference_results"
BATCH_SIZE=512
THRESHOLD=0.5
SEED=42

# 记录总开始时间（使用纳秒级精度）
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Dataset: ${DATA_PATH}"
echo "Model: ${MODEL_PATH}"
echo "========================================"

# 检查必要的文件
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 数据文件 $DATA_PATH 不存在"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "错误: 模型文件 $MODEL_PATH 不存在"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 初始化变量
INFERENCE_TIME=""
INFERENCE_MEMORY=""

# 运行推理并测量
echo ""
echo ">>> Running Inference..."

# 使用 /usr/bin/time 测量（只用于内存，不用它的时间）
if command -v /usr/bin/time >/dev/null 2>&1; then
    # 创建临时文件存储内存信息
    temp_time_file=$(mktemp)
    temp_output=$(mktemp)

    # 使用date命令精确计时
    PY_START_TIME=$(date +%s.%N)
    
    # 执行推理命令
    /usr/bin/time -f "MEMORY_KB:%M" -o "$temp_time_file" python infer_split_cpu.py \
        --data_path $DATA_PATH \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --threshold $THRESHOLD \
        --seed $SEED > "$temp_output" 2>&1
    EXIT_CODE=$?
    
    PY_END_TIME=$(date +%s.%N)
    
    # 显示命令输出
    cat "$temp_output"
    rm -f "$temp_output"

    # 使用date命令计算精确时间（与total time同样的计算方法）
    INFERENCE_TIME=$(echo "$PY_END_TIME $PY_START_TIME" | awk '{printf "%.6f", $1 - $2}' | sed 's/0*$//;s/\.$//')
    # 如果没有小数部分，补一个.0
    if [[ "$INFERENCE_TIME" != *.* ]]; then
        INFERENCE_TIME="${INFERENCE_TIME}.0"
    fi

    # 提取内存使用
    memory_kb=$(grep "MEMORY_KB:" "$temp_time_file" | cut -d: -f2)
    if [ -n "$memory_kb" ] && echo "$memory_kb" | grep -q "^[0-9]\+$"; then
        INFERENCE_MEMORY=$(echo "$memory_kb 1024" | awk '{printf "%.4f", $1/$2}')
    else
        INFERENCE_MEMORY="N/A"
    fi

    rm -f "$temp_time_file"
else
    # 如果没有 /usr/bin/time，只测量时间
    PY_START_TIME=$(date +%s.%N)
    python infer_split_cpu.py \
        --data_path $DATA_PATH \
        --model_path $MODEL_PATH \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --threshold $THRESHOLD \
        --seed $SEED
    EXIT_CODE=$?
    PY_END_TIME=$(date +%s.%N)
    
    # 计算时间差，保留原始精度
    INFERENCE_TIME=$(echo "$PY_END_TIME $PY_START_TIME" | awk '{printf "%.6f", $1 - $2}' | sed 's/0*$//;s/\.$//')
    if [[ "$INFERENCE_TIME" != *.* ]]; then
        INFERENCE_TIME="${INFERENCE_TIME}.0"
    fi
    INFERENCE_MEMORY="N/A"
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Inference failed with exit code $EXIT_CODE"
    exit 1
fi

echo "Inference completed in $INFERENCE_TIME seconds"
if [ "$INFERENCE_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${INFERENCE_MEMORY} MB"
fi

# ========== Summary ==========
TOTAL_END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$TOTAL_END_TIME $TOTAL_START_TIME" | awk '{printf "%.6f", $1 - $2}' | sed 's/0*$//;s/\.$//')
if [[ "$TOTAL_TIME" != *.* ]]; then
    TOTAL_TIME="${TOTAL_TIME}.0"
fi

echo ""
echo "========================================"
echo "Execution Summary"
echo "========================================"
printf "%-35s %-12s %-20s\n" "Step" "Time(s)" "Peak Memory(MB)"
echo "------------------------------------------------------------"

# 格式化输出，保留实际精度
if [ "$INFERENCE_TIME" != "N/A" ] && [ "$INFERENCE_MEMORY" != "N/A" ]; then
    printf "%-35s %-12s %-20s\n" "Inference" "$INFERENCE_TIME" "$INFERENCE_MEMORY"
elif [ "$INFERENCE_TIME" != "N/A" ]; then
    printf "%-35s %-12s %-20s\n" "Inference" "$INFERENCE_TIME" "N/A"
else
    printf "%-35s %-12s %-20s\n" "Inference" "N/A" "$INFERENCE_MEMORY"
fi

echo "------------------------------------------------------------"
echo "Total Time: $TOTAL_TIME seconds"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "推理完成！结果保存在: $OUTPUT_DIR"

# 检查 /usr/bin/time 是否可用
if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo ""
    echo "注意: /usr/bin/time 不可用，内存测量不可用"
    echo "如需内存测量，请安装: apt-get update && apt-get install -y time"
fi
