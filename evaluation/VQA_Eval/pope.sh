#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
 
CKPT="Liquid_V1_7B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader \
        --model-path  /path/to/Liquid_models//$CKPT \
        --question-file /path/to/eval/pope/llava_pope_test.jsonl \
        --image-folder /path/to/eval/pope/val2014  \
        --answers-file ./work_dirs/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode gemma &
done

wait

output_file=./work_dirs/pope/answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./work_dirs/pope/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python eval_pope.py \
    --annotation-dir /path/to/eval/pope/coco \
    --question-file /path/to/eval/pope/llava_pope_test.jsonl \
    --result-file $output_file
echo $CKPT