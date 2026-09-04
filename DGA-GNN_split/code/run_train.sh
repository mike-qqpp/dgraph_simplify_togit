# 同时指定模型保存路径和batch_size
#CUDA_VISIBLE_DEVICES=0 \
#python train_split.py --config-name dgraphfin \
#--checkpoint_path ../models/DGA-GNN/dgraphfin/model.pt \
#--batch_size 512



python train_split.py --config-name amazon \
--checkpoint_path ../models/DGA-GNN/amazon/model.pt \
--cpu \
--batch_size 512

