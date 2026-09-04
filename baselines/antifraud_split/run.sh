CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --method hogrl \
   #2>&1 | tee debug_output.txt


