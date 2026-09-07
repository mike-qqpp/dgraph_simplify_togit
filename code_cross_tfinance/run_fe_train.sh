#!/bin/bash

# Run all feature-engineering stages with parameters.
# Usage: ./run_features.sh <input_dir> <datatype> <output_dir>

# Set default values.
INPUT_DIR=${1:-"../data_split"}
DATATYPE=${2:-"train"}
OUTPUT_DIR=${3:-"../feature_split"}
DATANAME=${4:-"tfinance"}

echo "Starting T-Finance feature engineering..."
echo "Input directory: ${INPUT_DIR}"
echo "Data type: ${DATATYPE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Dataset name: ${DATANAME}"

# Create the output directory, including nested directories.
mkdir -p ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}

# 1. Base features
echo "Running base features..."
python 01_base_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/01_base_features.pkl

# 2. Structural features
echo "Running structural features..."
python 02_structural_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/02_structural_features.pkl

# 3. Neighbor features
echo "Running neighbor features..."
# python 03_neighbor_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/03_neighbor_features.pkl --max_neighbors_1hop 50 --max_neighbors_2hop 500
python 03_neighbor_features_v2.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/03_neighbor_features.pkl --max_neighbors_1hop 100 --max_neighbors_2hop 1000

# 4. Spectral features
echo "Running spectral features..."
python 04_spectral_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/04_spectral_features.pkl

# 5. Graph embedding features
echo "Running graph embedding features..."
python 05_embedding_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/05_embedding_features.pkl --embedding_dim 16

# 6. Clustering features
echo "Running clustering features..."
python 06_clustering_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/06_clustering_features.pkl

# 7. Anomaly-detection features
echo "Running anomaly-detection features..."
python 07_anomaly_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/07_anomaly_features.pkl

# 8. Mixed high-order features
echo "Running mixed high-order features..."
python 08_mixed_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/08_mixed_features.pkl

# 9. Advanced financial features
echo "Running advanced financial features..."
python 09_advanced_financial_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/09_advanced_financial_features.pkl 

echo "All feature-engineering stages are complete."
echo "Output files:"
ls -lh ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/*.pkl

# Run derived feature engineering.
echo "Running derived feature engineering..."
python 21_derived_features.py \
    --feature_paths ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/01_base_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/02_structural_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/03_neighbor_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/04_spectral_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/05_embedding_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/06_clustering_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/07_anomaly_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/08_mixed_features.pkl \
                    ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/09_advanced_financial_features.pkl \
    --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/21_derived_features.pkl

echo "All feature-engineering stages have finished."



