#!/bin/bash

# Read the dataset name and split from command-line arguments.
# Usage: ./run_fe_test.sh yelpchi test
DATASET=${1:-yelpchi}  # Default: yelpchi
SPLIT=${2:-test}      # Default: test

# Record the overall start time with nanosecond precision.
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Dataset: ${DATASET}"
echo "Split: ${SPLIT}"
echo "========================================"

# Check the required directories and files.
if [ ! -d "../data_split" ]; then
    echo "Error: directory ../data_split does not exist"
    exit 1
fi

if [ ! -f "../data_split/${DATASET}_${SPLIT}.npz" ]; then
    echo "Error: data file ../data_split/${DATASET}_${SPLIT}.npz does not exist"
    exit 1
fi

# Check whether the model directory exists.
if [ ! -d "../models_ours/${DATASET}" ]; then
    echo "Warning: directory ../models_ours/${DATASET} does not exist; testing may fail"
fi

# Create the feature output directory if it does not exist.
mkdir -p "../feature_split/${DATASET}/${SPLIT}"

# Initialize variables.
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

# Run a command and measure its time and memory usage.
run_with_measurement() {
    local step_name="$1"
    shift
    local cmd="$@"
    
    echo ""
    echo ">>> Running $step_name..."
    local START_TIME=$(date +%s.%N)
    
    # Use /usr/bin/time for memory measurement when available.
    if command -v /usr/bin/time >/dev/null 2>&1; then
        # Create a temporary file for timing and memory information.
        local temp_time_file=$(mktemp)
        
        # Run the command and capture memory usage with /usr/bin/time.
        local temp_output=$(mktemp)
        /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$temp_time_file" bash -c "$cmd" > "$temp_output" 2>&1
        local exit_code=$?
        
        # Display the command output.
        cat "$temp_output"
        rm -f "$temp_output"
        
        local END_TIME=$(date +%s.%N)
        
        # Extract memory usage.
        local memory_kb=$(grep "MEMORY_KB:" "$temp_time_file" | cut -d: -f2)
        local peak_memory_mb="N/A"
        if [ -n "$memory_kb" ] && echo "$memory_kb" | grep -q "^[0-9]\+$"; then
            peak_memory_mb=$(echo "$memory_kb 1024" | awk '{printf "%.4f", $1/$2}')
        fi
        
        # Calculate execution time.
        local execution_time=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')
        
        rm -f "$temp_time_file"
    else
        # Measure only time when /usr/bin/time is unavailable.
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
    
    # Return time and memory usage.
    echo "$execution_time $peak_memory_mb"
}

# Run all stages.
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

# Measure memory usage.
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
python train.py --dataset ${DATASET} \
  --data_dir ../feature_split \
  --data_split_dir ../data_split \
--output_dir ../models_ours

EXIT_CODE=$?
END_TIME=$(date +%s.%N)
TRAIN_TIME=$(echo "$END_TIME $START_TIME" | awk '{printf "%.4f", $1 - $2}')

if command -v /usr/bin/time >/dev/null 2>&1; then
    TEST_MEMORY_KB=$(/usr/bin/time -f "%M" python test.py --dataset ${DATASET} \
      --data_dir ../feature_split \
      --data_split_dir ../data_split \
      --model_dir ../models_ours 2>&1 >/dev/null | tail -1)
    if echo "$TEST_MEMORY_KB" | grep -q "^[0-9]\+$"; then
        TEST_MEMORY=$(echo "$TEST_MEMORY_KB 1024" | awk '{printf "%.4f", $1/$2}')
    else
        TEST_MEMORY="N/A"
    fi
else
    TEST_MEMORY="N/A"
fi

echo "Test completed in $TRAIN_TIME seconds"
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
printf "%-35s %-12s %-20s\n" "Train (Model Inference)" "$TRAIN_TIME" "$TEST_MEMORY"
echo "------------------------------------------------------------"

# Calculate the sum of stage runtimes.
PARTS_SUM=$(echo "$PART1_TIME $PART2_TIME $PART3_TIME $PARTMK_TIME $TRAIN_TIME" | awk '{printf "%.4f", $1+$2+$3+$4+$5}')

echo "Sum of parts: $PARTS_SUM seconds"
echo "Total Time: $TOTAL_TIME seconds"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Check whether /usr/bin/time is available.
if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo ""
    echo "Note: /usr/bin/time is unavailable; memory measurement is disabled"
    echo "Install it with: apt-get update && apt-get install -y time"
fi

echo ""
echo "========================================"
echo "Testing complete. Dataset: $DATASET, split: $SPLIT"
echo "Test results are saved in: ../models_ours/${DATASET}/"
echo "========================================"
