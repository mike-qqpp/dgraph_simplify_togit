#!/bin/bash

# 一键执行所有特征工程（参数化版本）
# 使用方法：./run_features.sh <input_dir> <datatype> <output_dir>

# 设置默认值
INPUT_DIR=${1:-"../data_split"}
DATATYPE=${2:-"test"}
OUTPUT_DIR=${3:-"../feature_split"}
DATANAME=${4:-"tfinance"}

echo "开始执行T-Finance特征工程..."
echo "输入目录: ${INPUT_DIR}"
echo "数据类型: ${DATATYPE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "数据集名称: ${DATANAME}"

# 创建输出目录（包括嵌套目录）
mkdir -p ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}

# 1. 基础特征
echo "执行基础特征..."
python 01_base_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/01_base_features.pkl

# 2. 结构特征
echo "执行结构特征..."
python 02_structural_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/02_structural_features.pkl

# 3. 邻居特征
echo "执行邻居特征..."
# python 03_neighbor_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/03_neighbor_features.pkl --max_neighbors_1hop 50 --max_neighbors_2hop 500
python 03_neighbor_features_v2.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/03_neighbor_features.pkl --max_neighbors_1hop 100 --max_neighbors_2hop 1000

# 4. 谱特征
echo "执行谱特征..."
python 04_spectral_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/04_spectral_features.pkl

# 5. 图嵌入特征
echo "执行图嵌入特征..."
python 05_embedding_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/05_embedding_features.pkl --embedding_dim 16

# 6. 聚类特征
echo "执行聚类特征..."
python 06_clustering_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/06_clustering_features.pkl

# 7. 异常检测特征
echo "执行异常检测特征..."
python 07_anomaly_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/07_anomaly_features.pkl

# 8. 混合高阶特征
echo "执行混合高阶特征..."
python 08_mixed_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/08_mixed_features.pkl

# 9. 高级金融特征
echo "执行高阶特征..."
python 09_advanced_financial_features.py --input_path ${INPUT_DIR}/${DATANAME}_${DATATYPE}.npz --output_path ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/09_advanced_financial_features.pkl 

echo "所有特征工程完成！"
echo "输出文件列表："
ls -lh ${OUTPUT_DIR}/${DATANAME}/${DATATYPE}/*.pkl

# 执行衍生特征工程
echo "执行衍生特征工程..."
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

echo "所有特征工程执行完毕！"







python test.py \
--dataset tfinance \
--data_dir ../feature_split \
--model_dir ../models/