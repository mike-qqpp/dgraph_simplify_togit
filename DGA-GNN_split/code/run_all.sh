#python train.py --config-name elliptic_of_amnet
#CUDA_VISIBLE_DEVICES=1 \
#       	python train_mk.py --config-name tfinance_mk

# CUDA_VISIBLE_DEVICES=1 \
#        python train_mk.py --config-name dgraphfin


#python train.py --config-name yelpchi
#python train.py --config-name amazon
#python train.py --config-name tsocial

#python data_handle_dgraph.py \
#	--file_input dgraph_exp130_v2_nona.npz \
#	--file_output dgraph_exp130_v2.dgldata


#CUDA_VISIBLE_DEVICES=1 \
#       python train_mk.py --config-name dgraph_exp130


#python data_handle_dgraph.py \
#        --file_input tfinance.npz \
#        --file_output tfinance.dgldata


CUDA_VISIBLE_DEVICES=0 \
       python train_mk_gpu.py --config-name tfinance_mk  #dgraphfin




