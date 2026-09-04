#python main_mk.py --dataset tfinance

#python makedata.py --dataste dgraphfin
#python main_mk.py --dataset dgraphfin

#python makedata.py --dataset  dgraph_exp130_v2_nona


CUDA_VISIBLE_DEVICES=0 \
python main_mk_v2.py --dataset  dgraph_exp130_v2_nona --data_path ../data --epoch 20 --run 1 \
	--train_batch_size 512 \
	--train_truncate 5000 \

#python main_mk_v2.py --dataset dgraph_exp130_nona --data_path ../data --epoch 100




#python main_mk_v2.py --dataset dgraphfin --data_path ../data --epoch 100

