#!/bin/bash

# 从命令行参数获取方法、数据集和模型目录
# 使用方法: ./run_infer_benchmark.sh hogrl dgraphfin ../models/hogrl/dgraphfin/dgraphfin
METHOD=${1:-rgtan}           # 默认值为hogrl
DATASET=${2:-dgraphfin}      # 默认值为dgraphfin
MODEL_DIR=${3:-../models/rgtan/dgraphfin/dgraphfin}  # 默认模型路径

# 记录总开始时间（使用纳秒级精度）
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Method: ${METHOD}"
echo "Dataset: ${DATASET}"
echo "Model directory: ${MODEL_DIR}"
echo "========================================"

# 检查必要的目录和文件
if [ ! -d "${MODEL_DIR}" ]; then
    echo "错误: 模型目录 ${MODEL_DIR} 不存在"
    exit 1
fi

# 初始化变量
LOADER_CREATION_TIME=""
LOADER_CREATION_MEMORY=""
MODEL_LOAD_TIME=""
MODEL_LOAD_MEMORY=""
INFERENCE_TIME=""
INFERENCE_MEMORY=""
TOTAL_TIME=""
TOTAL_MEMORY=""

# 检查 /usr/bin/time 是否可用
TIME_AVAILABLE=false
if command -v /usr/bin/time >/dev/null 2>&1; then
    TIME_AVAILABLE=true
    echo "✓ /usr/bin/time 可用，将测量内存使用"
else
    echo "⚠️ /usr/bin/time 不可用，内存测量将显示为N/A"
    echo "   如需内存测量，请安装: sudo apt-get update && sudo apt-get install -y time"
fi
echo ""

# 函数：运行命令并测量时间和内存
run_with_measurement() {
    local step_name="$1"
    shift
    local cmd="$@"
    
    echo ">>> Running $step_name..."
    local START_TIME=$(date +%s.%N)
    
    local peak_memory_mb="N/A"
    
    if [ "$TIME_AVAILABLE" = true ]; then
        # 创建临时文件存储时间和内存信息
        local temp_time_file=$(mktemp)
        local temp_output=$(mktemp)
        
        # 使用 /usr/bin/time 执行命令
        /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$temp_time_file" bash -c "$cmd" > "$temp_output" 2>&1
        local exit_code=$?
        
        # 显示命令输出
        cat "$temp_output"
        rm -f "$temp_output"
        
        # 提取内存使用
        if [ -f "$temp_time_file" ]; then
            local memory_kb=$(grep "MEMORY_KB:" "$temp_time_file" | cut -d: -f2 | tr -d ' ')
            if [ -n "$memory_kb" ] && [[ "$memory_kb" =~ ^[0-9]+$ ]]; then
                peak_memory_mb=$(echo "scale=2; $memory_kb / 1024" | bc 2>/dev/null)
                if [ -z "$peak_memory_mb" ]; then
                    peak_memory_mb=$(awk "BEGIN {printf \"%.2f\", $memory_kb/1024}" 2>/dev/null)
                fi
            fi
            rm -f "$temp_time_file"
        fi
    else
        # 如果没有 /usr/bin/time，只执行命令
        eval "$cmd"
        local exit_code=$?
    fi
    
    local END_TIME=$(date +%s.%N)
    local execution_time=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null)
    if [ -z "$execution_time" ]; then
        execution_time=$(awk "BEGIN {printf \"%.4f\", $END_TIME - $START_TIME}" 2>/dev/null)
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

# 运行主推理
echo ""
echo ">>> Running Main Inference..."

# 构建命令
CMD="python main_split_cpu.py --method ${METHOD} --mode infer --model_dir ${MODEL_DIR}"

START_TIME=$(date +%s.%N)

if [ "$TIME_AVAILABLE" = true ]; then
    # 创建临时文件
    temp_time_file=$(mktemp)
    temp_output=$(mktemp)
    
    # 使用 /usr/bin/time 执行
    /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$temp_time_file" $CMD > "$temp_output" 2>&1
    exit_code=$?
    
    # 显示输出
    cat "$temp_output"
    rm -f "$temp_output"
    
    # 提取内存使用
    if [ -f "$temp_time_file" ]; then
        memory_kb=$(grep "MEMORY_KB:" "$temp_time_file" | cut -d: -f2 | tr -d ' ')
        if [ -n "$memory_kb" ] && [[ "$memory_kb" =~ ^[0-9]+$ ]]; then
            TOTAL_MEMORY=$(echo "scale=2; $memory_kb / 1024" | bc 2>/dev/null)
            if [ -z "$TOTAL_MEMORY" ]; then
                TOTAL_MEMORY=$(awk "BEGIN {printf \"%.2f\", $memory_kb/1024}" 2>/dev/null)
            fi
        else
            TOTAL_MEMORY="N/A"
        fi
        rm -f "$temp_time_file"
    else
        TOTAL_MEMORY="N/A"
    fi
else
    # 直接执行
    $CMD
    exit_code=$?
    TOTAL_MEMORY="N/A"
fi

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc 2>/dev/null)
if [ -z "$TOTAL_TIME" ]; then
    TOTAL_TIME=$(awk "BEGIN {printf \"%.4f\", $END_TIME - $START_TIME}" 2>/dev/null)
fi

echo ""
echo "Main inference completed in $TOTAL_TIME seconds"
if [ "$TOTAL_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${TOTAL_MEMORY} MB"
fi

# ========== Summary ==========
TOTAL_END_TIME=$(date +%s.%N)
TOTAL_WALL_TIME=$(echo "$TOTAL_END_TIME - $TOTAL_START_TIME" | bc 2>/dev/null)
if [ -z "$TOTAL_WALL_TIME" ]; then
    TOTAL_WALL_TIME=$(awk "BEGIN {printf \"%.4f\", $TOTAL_END_TIME - $TOTAL_START_TIME}" 2>/dev/null)
fi

echo ""
echo "========================================"
echo "Execution Summary"
echo "========================================"
printf "%-35s %-12s %-20s\n" "Step" "Time(s)" "Peak Memory(MB)"
echo "------------------------------------------------------------"
printf "%-35s %-12s %-20s\n" "Main Inference" "$TOTAL_TIME" "$TOTAL_MEMORY"
echo "------------------------------------------------------------"
echo "Total Wall Time: $TOTAL_WALL_TIME seconds"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 检查 /usr/bin/time 是否可用
if [ "$TIME_AVAILABLE" = false ]; then
    echo ""
    echo "注意: /usr/bin/time 不可用，内存测量不可用"
    echo "如需内存测量，请安装: sudo apt-get update && sudo apt-get install -y time"
fi
