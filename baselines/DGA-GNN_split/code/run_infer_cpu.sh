#!/bin/bash

# 从命令行参数获取数据集名称和模型路径
# 使用方法: ./run_infer_cpu.sh amazon
DATASET=${1:-dgraphfin}  # 默认值为amazon

# 记录总开始时间（使用纳秒级精度）
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Dataset: ${DATASET}"
echo "Mode: CPU Only Inference"
echo "========================================"

# 定义模型路径
MODEL_PATH="../models/DGA-GNN/${DATASET}/model.pt"
BATCH_SIZE=512

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误: 模型文件 ${MODEL_PATH} 不存在"
    exit 1
fi

# 初始化变量
INFER_TIME=""
INFER_MEMORY=""

# 函数：运行命令并测量时间和内存
run_with_measurement() {
    local step_name="$1"
    shift
    local cmd="$@"
    
    echo ""
    echo ">>> Running $step_name..."
    local START_TIME=$(date +%s.%N)
    
    # 如果有 /usr/bin/time，使用它来测量内存
    if command -v /usr/bin/time >/dev/null 2>&1; then
        # 创建临时文件存储时间和内存信息
        local temp_time_file=$(mktemp)
        
        # 执行命令，使用 /usr/bin/time 捕获内存
        local temp_output=$(mktemp)
        /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$temp_time_file" bash -c "$cmd" > "$temp_output" 2>&1
        local exit_code=$?
        
        # 显示命令输出
        cat "$temp_output"
        rm -f "$temp_output"
        
        local END_TIME=$(date +%s.%N)
        
        # 提取内存使用
        local memory_kb=$(grep "MEMORY_KB:" "$temp_time_file" | cut -d: -f2)
        local peak_memory_mb="N/A"
        if [ -n "$memory_kb" ] && echo "$memory_kb" | grep -q "^[0-9]\+$"; then
            peak_memory_mb=$(echo "$memory_kb 1024" | awk '{printf "%.4f", $1/$2}')
        fi
        
        # 计算执行时间
        local execution_time=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')
        
        rm -f "$temp_time_file"
    else
        # 如果没有 /usr/bin/time，只测量时间
        eval "$cmd"
        local exit_code=$?
        local END_TIME=$(date +%s.%N)
        
        local execution_time=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')
        local peak_memory_mb="N/A"
    fi
    
    if [ $exit_code -ne 0 ]; then
        echo "Error: $step_name failed with exit code $exit_code"
        exit 1
    fi
    
    echo "$step_name completed in $execution_time seconds"
    if [ "$peak_memory_mb" != "N/A" ]; then
        echo "  Peak memory usage: ${peak_memory_mb} MB"
    fi
    
    # 返回时间和内存
    echo "$execution_time $peak_memory_mb"
}

echo ""

# ========== Run Inference ==========
echo ">>> Running Inference with DGA Model (CPU Mode)..."
START_TIME=$(date +%s.%N)

# 构建命令
CMD="python infer_split_cpu.py --config-name ${DATASET} --model_path ${MODEL_PATH} --batch_size ${BATCH_SIZE}"

# 如果有 /usr/bin/time，使用它来测量内存
if command -v /usr/bin/time >/dev/null 2>&1; then
    # 创建临时文件存储时间和内存信息
    temp_time_file=$(mktemp)
    
    # 执行命令，使用 /usr/bin/time 捕获内存
    temp_output=$(mktemp)
    /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$temp_time_file" bash -c "$CMD" > "$temp_output" 2>&1
    exit_code=$?
    
    # 显示命令输出
    cat "$temp_output"
    rm -f "$temp_output"
    
    END_TIME=$(date +%s.%N)
    
    # 提取内存使用
    memory_kb=$(grep "MEMORY_KB:" "$temp_time_file" | cut -d: -f2)
    if [ -n "$memory_kb" ] && echo "$memory_kb" | grep -q "^[0-9]\+$"; then
        INFER_MEMORY=$(echo "$memory_kb 1024" | awk '{printf "%.4f", $1/$2}')
    else
        INFER_MEMORY="N/A"
    fi
    
    rm -f "$temp_time_file"
else
    # 如果没有 /usr/bin/time，只测量时间
    eval "$CMD"
    exit_code=$?
    END_TIME=$(date +%s.%N)
    INFER_MEMORY="N/A"
fi

INFER_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

echo ""
echo "Inference completed in $INFER_TIME seconds"
if [ "$INFER_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${INFER_MEMORY} MB"
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
printf "%-35s %-12s %-20s\n" "DGA Model Inference (CPU)" "$INFER_TIME" "$INFER_MEMORY"
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
