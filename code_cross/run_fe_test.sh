#!/bin/bash

# 从命令行参数获取数据集名称和划分
# 使用方法: ./run_fe_test.sh yelpchi test
DATASET=${1:-yelpchi}  # 默认值为yelpchi
SPLIT=${2:-test}      # 默认值为test

# 记录总开始时间（使用纳秒级精度）
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Dataset: ${DATASET}"
echo "Split: ${SPLIT}"
echo "========================================"

# 检查必要的目录和文件
if [ ! -d "../data_split" ]; then
    echo "错误: ../data_split 目录不存在"
    exit 1
fi

if [ ! -f "../data_split/${DATASET}_${SPLIT}.npz" ]; then
    echo "错误: 数据文件 ../data_split/${DATASET}_${SPLIT}.npz 不存在"
    exit 1
fi

# 创建特征保存目录（如果不存在）
mkdir -p "../feature_split/${DATASET}/${SPLIT}"

# 初始化变量
PART1_TIME=""
PART1_MEMORY=""
PART2_TIME=""
PART2_MEMORY=""
PART3_TIME=""
PART3_MEMORY=""
PARTMK_TIME=""
PARTMK_MEMORY=""
TEST_TIME=""
TEST_MEMORY=""

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

# 运行各步骤
echo ""

# ========== Part 1 ==========
echo ">>> Running Part 1: Feature Extraction..."
START_TIME=$(date +%s.%N)
python runfe_part1_new.py \
  --path_data ../data_split/${DATASET}_${SPLIT}.npz \
  --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature_new.pkl
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
PART1_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

# 测量内存
if command -v /usr/bin/time >/dev/null 2>&1; then
    PART1_MEMORY_KB=$(/usr/bin/time -f "%M" python runfe_part1_new.py \
      --path_data ../data_split/${DATASET}_${SPLIT}.npz \
      --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature_new.pkl 2>&1 >/dev/null | tail -1)
    if echo "$PART1_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        PART1_MEMORY=$(echo "$PART1_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        PART1_MEMORY="N/A"
    fi
else
    PART1_MEMORY="N/A"
fi

echo "Part 1 completed in $PART1_TIME seconds"
if [ "$PART1_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${PART1_MEMORY} MB"
fi

# ========== Part 2 ==========
echo ""
echo ">>> Running Part 2: Feature Processing..."
START_TIME=$(date +%s.%N)
python runfe_part2.py \
  --path_data ../data_split/${DATASET}_${SPLIT}.npz \
  --path_bin_dict ../feature_split/${DATASET}/${SPLIT}/bin_dict.pkl \
  --path_bin_prob_dict ../feature_split/${DATASET}/${SPLIT}/x.pkl \
  --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature.pkl \
  --path_save_feature_supplement ../feature_split/${DATASET}/${SPLIT}/feature_supplement.pkl \
  --path_save_feature_null ../feature_split/${DATASET}/${SPLIT}/feature_null.pkl \
  --path_save_feature_mk ../feature_split/${DATASET}/${SPLIT}/df_node_enhanced_mk.pkl
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
PART2_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

if command -v /usr/bin/time >/dev/null 2>&1; then
    PART2_MEMORY_KB=$(/usr/bin/time -f "%M" python runfe_part2.py \
      --path_data ../data_split/${DATASET}_${SPLIT}.npz \
      --path_bin_dict ../feature_split/${DATASET}/${SPLIT}/bin_dict.pkl \
      --path_bin_prob_dict ../feature_split/${DATASET}/${SPLIT}/x.pkl \
      --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature.pkl \
      --path_save_feature_supplement ../feature_split/${DATASET}/${SPLIT}/feature_supplement.pkl \
      --path_save_feature_null ../feature_split/${DATASET}/${SPLIT}/feature_null.pkl \
      --path_save_feature_mk ../feature_split/${DATASET}/${SPLIT}/df_node_enhanced_mk.pkl 2>&1 >/dev/null | tail -1)
    if echo "$PART2_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        PART2_MEMORY=$(echo "$PART2_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        PART2_MEMORY="N/A"
    fi
else
    PART2_MEMORY="N/A"
fi

echo "Part 2 completed in $PART2_TIME seconds"
if [ "$PART2_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${PART2_MEMORY} MB"
fi

# ========== Part 3 ==========
echo ""
echo ">>> Running Part 3: Feature Enhancement..."
START_TIME=$(date +%s.%N)
python runfe_part3.py \
  --path_data ../data_split/${DATASET}_${SPLIT}.npz \
  --path_bin_dict ../feature_split/${DATASET}/${SPLIT}/bin_dict.pkl \
  --path_bin_prob_dict ../feature_split/${DATASET}/${SPLIT}/x.pkl \
  --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature.pkl \
  --path_save_feature_supplement ../feature_split/${DATASET}/${SPLIT}/feature_supplement.pkl \
  --path_save_feature_null ../feature_split/${DATASET}/${SPLIT}/feature_null.pkl \
  --path_save_feature_mk ../feature_split/${DATASET}/${SPLIT}/df_node_enhanced_mk.pkl
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
PART3_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

if command -v /usr/bin/time >/dev/null 2>&1; then
    PART3_MEMORY_KB=$(/usr/bin/time -f "%M" python runfe_part3.py \
      --path_data ../data_split/${DATASET}_${SPLIT}.npz \
      --path_bin_dict ../feature_split/${DATASET}/${SPLIT}/bin_dict.pkl \
      --path_bin_prob_dict ../feature_split/${DATASET}/${SPLIT}/x.pkl \
      --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature.pkl \
      --path_save_feature_supplement ../feature_split/${DATASET}/${SPLIT}/feature_supplement.pkl \
      --path_save_feature_null ../feature_split/${DATASET}/${SPLIT}/feature_null.pkl \
      --path_save_feature_mk ../feature_split/${DATASET}/${SPLIT}/df_node_enhanced_mk.pkl 2>&1 >/dev/null | tail -1)
    if echo "$PART3_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        PART3_MEMORY=$(echo "$PART3_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        PART3_MEMORY="N/A"
    fi
else
    PART3_MEMORY="N/A"
fi

echo "Part 3 completed in $PART3_TIME seconds"
if [ "$PART3_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${PART3_MEMORY} MB"
fi

# ========== Part MK ==========
echo ""
echo ">>> Running Part MK: Feature MK..."
START_TIME=$(date +%s.%N)
python runfe_mk.py \
  --path_data ../data_split/${DATASET}_${SPLIT}.npz \
  --path_bin_dict ../feature_split/${DATASET}/${SPLIT}/bin_dict.pkl \
  --path_bin_prob_dict ../feature_split/${DATASET}/${SPLIT}/x.pkl \
  --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature.pkl \
  --path_save_feature_supplement ../feature_split/${DATASET}/${SPLIT}/feature_supplement.pkl \
  --path_save_feature_null ../feature_split/${DATASET}/${SPLIT}/feature_null.pkl \
  --path_save_feature_mk ../feature_split/${DATASET}/${SPLIT}/df_node_enhanced_mk.pkl
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
PARTMK_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

if command -v /usr/bin/time >/dev/null 2>&1; then
    PARTMK_MEMORY_KB=$(/usr/bin/time -f "%M" python runfe_mk.py \
      --path_data ../data_split/${DATASET}_${SPLIT}.npz \
      --path_bin_dict ../feature_split/${DATASET}/${SPLIT}/bin_dict.pkl \
      --path_bin_prob_dict ../feature_split/${DATASET}/${SPLIT}/x.pkl \
      --path_save_feature ../feature_split/${DATASET}/${SPLIT}/feature.pkl \
      --path_save_feature_supplement ../feature_split/${DATASET}/${SPLIT}/feature_supplement.pkl \
      --path_save_feature_null ../feature_split/${DATASET}/${SPLIT}/feature_null.pkl \
      --path_save_feature_mk ../feature_split/${DATASET}/${SPLIT}/df_node_enhanced_mk.pkl 2>&1 >/dev/null | tail -1)
    if echo "$PARTMK_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        PARTMK_MEMORY=$(echo "$PARTMK_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        PARTMK_MEMORY="N/A"
    fi
else
    PARTMK_MEMORY="N/A"
fi

echo "Part MK completed in $PARTMK_TIME seconds"
if [ "$PARTMK_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${PARTMK_MEMORY} MB"
fi

# ========== Test ==========
echo ""
echo ">>> Running Test: Model Inference..."
START_TIME=$(date +%s.%N)
python test.py --dataset ${DATASET}
EXIT_CODE=$?
END_TIME=$(date +%s.%N)
TEST_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

if command -v /usr/bin/time >/dev/null 2>&1; then
    TEST_MEMORY_KB=$(/usr/bin/time -f "%M" python test.py 2>&1 >/dev/null | tail -1)
    if echo "$TEST_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        TEST_MEMORY=$(echo "$TEST_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        TEST_MEMORY="N/A"
    fi
else
    TEST_MEMORY="N/A"
fi

echo "Test completed in $TEST_TIME seconds"
if [ "$TEST_MEMORY" != "N/A" ]; then
    echo "  Peak memory usage: ${TEST_MEMORY} MB"
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
printf "%-35s %-12s %-20s\n" "Part 1 (Feature Extraction)" "$PART1_TIME" "$PART1_MEMORY"
printf "%-35s %-12s %-20s\n" "Part 2 (Feature Processing)" "$PART2_TIME" "$PART2_MEMORY"
printf "%-35s %-12s %-20s\n" "Part 3 (Feature Enhancement)" "$PART3_TIME" "$PART3_MEMORY"
printf "%-35s %-12s %-20s\n" "Part MK (Feature MK)" "$PARTMK_TIME" "$PARTMK_MEMORY"
printf "%-35s %-12s %-20s\n" "Test (Model Inference)" "$TEST_TIME" "$TEST_MEMORY"
echo "------------------------------------------------------------"

# 计算各部分时间之和
PARTS_SUM=$(echo "$PART1_TIME $PART2_TIME $PART3_TIME $PARTMK_TIME $TEST_TIME" | awk '{printf "%.4f", $1+$2+$3+$4+$5}')

echo "Sum of parts: $PARTS_SUM seconds"
echo "Total Time: $TOTAL_TIME seconds"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 检查 /usr/bin/time 是否可用
if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo ""
    echo "注意: /usr/bin/time 不可用，内存测量不可用"
    echo "如需内存测量，请安装: apt-get update && apt-get install -y time"
fi
