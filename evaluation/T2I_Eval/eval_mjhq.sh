#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
 
CKPT="/path/to/Liquid_models//Liquid_V1_7B/"
SAVE_PTH="./mjhq_results"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python mjhq_generation.py \
        --model_path $CKPT \
        --save_path $SAVE_PTH \
        --cfg_scale 7.0 \
        --load_8bit True \
        --batch_size 4 \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
done
wait
 
python get_fid.py --src_dir MJHQ-30K/mjhq30k_imgs --gen_dir $SAVE_PTH