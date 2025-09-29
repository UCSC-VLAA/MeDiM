#!/bin/bash
SPLIT="llava_vqav2_mscoco_test-dev2015"

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="Liquid_V1_7B"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m model_vqa_loader \
    --model-path  /path/to/Liquid_models//$CKPT \
    --question-file /path/to/eval/vqav2/$SPLIT.jsonl \
    --image-folder /path/to/eval/vqav2/test2015 \
    --answers-file /path/to/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --temperature 0 \
    --conv-mode gemma &
done

wait


output_file=/path/to/eval/vqav2/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /path/to/eval/vqav2/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir /path/to/eval/vqav2/


echo $CKPT