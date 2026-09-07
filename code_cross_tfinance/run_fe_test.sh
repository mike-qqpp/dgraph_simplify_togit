#!/bin/bash

# Read the dataset name, data type, and I/O directories from command-line arguments.
# Usage: ./run_features_benchmark.sh [dataset] [split] [input_dir] [output_dir]
DATASET=${1:-tfinance}   # Default: tfinance
SPLIT=${2:-test}        # Default: test
INPUT_DIR=${3:-../data_split}
OUTPUT_DIR=${4:-../feature_split}

# Record the overall start time.
TOTAL_START_TIME=$(date +%s.%N)
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Dataset: ${DATASET}"
echo "Split: ${SPLIT}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "========================================"

# Check the required directories and files.
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: directory ${INPUT_DIR} does not exist"
    exit 1
fi

if [ ! -f "${INPUT_DIR}/${DATASET}_${SPLIT}.npz" ]; then
    echo "Error: data file ${INPUT_DIR}/${DATASET}_${SPLIT}.npz does not exist"
    exit 1
fi

# Create the feature output directory.
mkdir -p "${OUTPUT_DIR}/${DATASET}/${SPLIT}"

# Initialize variables.
STEP1_TIME=""; STEP1_MEMORY=""
STEP2_TIME=""; STEP2_MEMORY=""
STEP3_TIME=""; STEP3_MEMORY=""
STEP4_TIME=""; STEP4_MEMORY=""
STEP5_TIME=""; STEP5_MEMORY=""
STEP6_TIME=""; STEP6_MEMORY=""
STEP7_TIME=""; STEP7_MEMORY=""
STEP8_TIME=""; STEP8_MEMORY=""
STEP9_TIME=""; STEP9_MEMORY=""
STEP21_TIME=""; STEP21_MEMORY=""
TEST_TIME=""; TEST_MEMORY=""

# Run a command while measuring time and memory in one execution.
run_command() {
    local step_name="$1"
    shift
    local cmd="$@"
    
    echo ""
    echo ">>> Running ${step_name}..."
    
    if command -v /usr/bin/time >/dev/null 2>&1; then
        # Store the time-command output in a temporary file.
        local time_file=$(mktemp)
        local output_file=$(mktemp)
        local start_time=$(date +%s.%N)
        
        # Run the command, capture time and memory, and save stdout and stderr.
        /usr/bin/time -f "TIME_REAL:%e\nMEMORY_KB:%M" -o "$time_file" bash -c "$cmd" > "$output_file" 2>&1
        local exit_code=$?
        local end_time=$(date +%s.%N)
        
        # Display the command output.
        cat "$output_file"
        
        # Read time and memory values.
        local exec_time=$(grep "TIME_REAL:" "$time_file" | cut -d: -f2)
        local mem_kb=$(grep "MEMORY_KB:" "$time_file" | cut -d: -f2)
        
        rm -f "$time_file" "$output_file"
        
        # Use start/end timestamps if no timing result is available.
        if [ -z "$exec_time" ]; then
            exec_time=$(echo "$end_time $start_time" | awk '{printf "%.4f", $1 - $2}')
        fi
        
        # Convert memory to MB.
        local mem_mb="N/A"
        if [ -n "$mem_kb" ] && echo "$mem_kb" | grep -q "^[0-9]\+$"; then
            mem_mb=$(echo "$mem_kb 1024" | awk '{printf "%.4f", $1/$2}')
        fi
        
        echo "${step_name} completed in ${exec_time} seconds"
        if [ "$mem_mb" != "N/A" ]; then
            echo "  Peak memory usage: ${mem_mb} MB"
        fi
        
        # Return time and memory with a separator that cannot be confused with output.
        echo "RUN_COMMAND_RESULT:${exec_time}|${mem_mb}"
    else
        # Measure only time when /usr/bin/time is unavailable.
        local output_file=$(mktemp)
        local start_time=$(date +%s.%N)
        bash -c "$cmd" > "$output_file" 2>&1
        local exit_code=$?
        local end_time=$(date +%s.%N)
        
        # Display the command output.
        cat "$output_file"
        rm -f "$output_file"
        
        local exec_time=$(echo "$end_time $start_time" | awk '{printf "%.4f", $1 - $2}')
        
        echo "${step_name} completed in ${exec_time} seconds"
        echo "RUN_COMMAND_RESULT:${exec_time}|N/A"
    fi
    
    if [ $exit_code -ne 0 ]; then
        echo "Error: ${step_name} failed with exit code $exit_code"
        exit 1
    fi
}

# ========== 1. Base Features ==========
echo ""
echo ">>> Running Part 1: Base Features..."
result=$(run_command "Part 1 (Base Features)" \
    "python 01_base_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/01_base_features.pkl'")
STEP1_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP1_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 2. Structural Features ==========
echo ""
echo ">>> Running Part 2: Structural Features..."
result=$(run_command "Part 2 (Structural Features)" \
    "python 02_structural_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/02_structural_features.pkl'")
STEP2_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP2_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 3. Neighbor Features ==========
echo ""
echo ">>> Running Part 3: Neighbor Features..."
result=$(run_command "Part 3 (Neighbor Features)" \
    "python 03_neighbor_features_v2.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/03_neighbor_features.pkl' --max_neighbors_1hop 100 --max_neighbors_2hop 1000")
STEP3_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP3_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 4. Spectral Features ==========
echo ""
echo ">>> Running Part 4: Spectral Features..."
result=$(run_command "Part 4 (Spectral Features)" \
    "python 04_spectral_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/04_spectral_features.pkl'")
STEP4_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP4_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 5. Graph Embedding Features ==========
echo ""
echo ">>> Running Part 5: Embedding Features..."
result=$(run_command "Part 5 (Embedding Features)" \
    "python 05_embedding_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/05_embedding_features.pkl' --embedding_dim 16")
STEP5_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP5_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 6. Clustering Features ==========
echo ""
echo ">>> Running Part 6: Clustering Features..."
result=$(run_command "Part 6 (Clustering Features)" \
    "python 06_clustering_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/06_clustering_features.pkl'")
STEP6_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP6_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 7. Anomaly-Detection Features ==========
echo ""
echo ">>> Running Part 7: Anomaly Features..."
result=$(run_command "Part 7 (Anomaly Features)" \
    "python 07_anomaly_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/07_anomaly_features.pkl'")
STEP7_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP7_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 8. Mixed High-Order Features ==========
echo ""
echo ">>> Running Part 8: Mixed Features..."
result=$(run_command "Part 8 (Mixed Features)" \
    "python 08_mixed_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/08_mixed_features.pkl'")
STEP8_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP8_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== 9. Advanced Financial Features ==========
echo ""
echo ">>> Running Part 9: Advanced Financial Features..."
result=$(run_command "Part 9 (Advanced Financial Features)" \
    "python 09_advanced_financial_features.py --input_path '${INPUT_DIR}/${DATASET}_${SPLIT}.npz' --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/09_advanced_financial_features.pkl'")
STEP9_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP9_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== List Generated Feature Files ==========
echo ""
echo ">>> Generated feature files:"
ls -lh "${OUTPUT_DIR}/${DATASET}/${SPLIT}"/*.pkl 2>/dev/null || echo "No feature files are available"

# ========== 21. Derived Feature Engineering ==========
echo ""
echo ">>> Running Part 21: Derived Features..."
result=$(run_command "Part 21 (Derived Features)" \
    "python 21_derived_features.py \
        --feature_paths \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/01_base_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/02_structural_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/03_neighbor_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/04_spectral_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/05_embedding_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/06_clustering_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/07_anomaly_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/08_mixed_features.pkl' \
            '${OUTPUT_DIR}/${DATASET}/${SPLIT}/09_advanced_financial_features.pkl' \
        --output_path '${OUTPUT_DIR}/${DATASET}/${SPLIT}/21_derived_features.pkl'")
STEP21_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
STEP21_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== Test: Model Inference ==========
echo ""
echo ">>> Running Test: Model Inference..."
result=$(run_command "Test (Model Inference)" \
    "python test.py --dataset '${DATASET}' --data_dir '${OUTPUT_DIR}' --model_dir ../models_ours/")
TEST_TIME=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f1)
TEST_MEMORY=$(echo "$result" | grep "RUN_COMMAND_RESULT:" | tail -1 | cut -d':' -f2- | cut -d'|' -f2)

# ========== Final Summary ==========
TOTAL_END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$TOTAL_END_TIME $TOTAL_START_TIME" | awk '{printf "%.4f", $1 - $2}')

echo ""
echo "========================================"
echo "Execution Summary"
echo "========================================"
printf "%-42s %-12s %-20s\n" "Step" "Time(s)" "Peak Memory(MB)"
echo "------------------------------------------------------------"
printf "%-42s %-12s %-20s\n" "Part 1  (Base Features)"            "${STEP1_TIME:-0}"    "${STEP1_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 2  (Structural Features)"      "${STEP2_TIME:-0}"    "${STEP2_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 3  (Neighbor Features)"        "${STEP3_TIME:-0}"    "${STEP3_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 4  (Spectral Features)"        "${STEP4_TIME:-0}"    "${STEP4_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 5  (Embedding Features)"       "${STEP5_TIME:-0}"    "${STEP5_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 6  (Clustering Features)"      "${STEP6_TIME:-0}"    "${STEP6_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 7  (Anomaly Features)"         "${STEP7_TIME:-0}"    "${STEP7_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 8  (Mixed Features)"           "${STEP8_TIME:-0}"    "${STEP8_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 9  (Advanced Financial Feat.)" "${STEP9_TIME:-0}"    "${STEP9_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Part 21 (Derived Features)"         "${STEP21_TIME:-0}"   "${STEP21_MEMORY:-N/A}"
printf "%-42s %-12s %-20s\n" "Test    (Model Inference)"          "${TEST_TIME:-0}"     "${TEST_MEMORY:-N/A}"
echo "------------------------------------------------------------"

# Calculate the sum of stage runtimes.
PARTS_SUM=$(echo "${STEP1_TIME:-0} ${STEP2_TIME:-0} ${STEP3_TIME:-0} ${STEP4_TIME:-0} ${STEP5_TIME:-0} ${STEP6_TIME:-0} ${STEP7_TIME:-0} ${STEP8_TIME:-0} ${STEP9_TIME:-0} ${STEP21_TIME:-0} ${TEST_TIME:-0}" | awk '{printf "%.4f", $1+$2+$3+$4+$5+$6+$7+$8+$9+$10+$11}')

echo "Sum of parts: ${PARTS_SUM} seconds"
echo "Total Time:   ${TOTAL_TIME} seconds"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Check whether /usr/bin/time is available.
if ! command -v /usr/bin/time >/dev/null 2>&1; then
    echo ""
    echo "Note: /usr/bin/time is unavailable; memory measurement is disabled"
    echo "Install it with: apt-get update && apt-get install -y time"
fi
