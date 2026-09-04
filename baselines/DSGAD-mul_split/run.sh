CUDA_VISIBLE_DEVICES=0 python t.py \
    --model DSGAD \
    --run 1 \
    --dataset dgraph \
    --epoch 1 \
    --batch_size 512 \
    --lr 1e-3 \
    --lr_weights 1e-1 \
    --patience 10 \
    --model_config '{"h_feats": 32, "d": 2, "mix_beta": 2, "use_reduced_memory": true}'
