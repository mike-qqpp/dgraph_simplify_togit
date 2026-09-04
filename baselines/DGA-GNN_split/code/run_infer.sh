CUDA_VISIBLE_DEVICES=0 \
python infer_split.py --config-name amazon \
--model_path ../models/DGA-GNN/amazon/model.pt \
--cpu \
--batch_size 512
