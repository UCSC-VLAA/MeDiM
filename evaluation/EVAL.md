# Tutorial for Evaluation

Liquid can be directly tested on text-to-image, visual understanding, language tasks. Since Liquid is essentially an extension of the vocabulary of various pre-existing LLMs, it does not require any specific environment and can run solely on the huggingface `transformers` library. For our training and inference processes, the environment we used is 
```bash
python=3.9
torch=2.1.0
flash-attention=2.5.8
transformers=4.39.2
```
 We recommend using a similar environment, especially the same `transformers` version for inference. However, other more updated version should also work fine.


# Preparation

```bash
git clone https://github.com/FoundationVision/Liquid.git
cd evaluation

# download our instructed 7B model from huggingface
huggingface-cli download    --resume-download Junfeng5/Liquid_V1_7B  --local-dir  /path/to/Liquid_V1_7B

# download VQVAE config and weight
wget -P chameleon/ https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt 
wget -P chameleon/ https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml

```

# Text-to-Image Evaluation

## GenAI-Bench

```bash
cd T2I_Eval

# download GenAI-Bench-527 prompts
wget https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527/resolve/main/prompts.txt
wget https://huggingface.co/datasets/zhiqiulin/GenAI-Bench-527/resolve/main/genai_skills.json


# run image generation
# For GPUs with less than 40GB of VRAM (such as the 3090 or 4090), it is recommended to set `load_8bit` to `True` and keep the `batch_size` below 4. For GPUs with larger VRAM, you can set `batch_size=16` to accelerate the process.
bash eval_genai.sh

# The calculation of the VQA-score metric requires a specific environment with `python>=3.10` and `transformers>=4.45.0`. A new environment needs to be created unless the existing inference environment already meets these requirements.
# follow install instruction in https://github.com/linzhiqiu/t2v_metrics
conda create -n t2v python=3.10 -y
conda activate t2v
conda install pip -y
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install t2v-metrics

# get VQA-scaore on GenAI-Bench
python eval_genai_527.py --image_dir $SAVE_PTH

```


## MJHQ-30K

```bash
cd T2I_Eval

# download MJHQ-30K datasets
huggingface-cli download  --repo-type dataset --resume-download playgroundai/MJHQ-30K  --local-dir  MJHQ-30K
cd MJHQ-30K
unzip -q mjhq30k_imgs.zip  -d  mjhq30k_imgs
cd ..

# install clean-fid
pip install clean-fid
pip install scipy==1.11.1

# Run 30k image generation. Generating 30k images requires a significant amount of time, and it is recommended to use a larger `batch_size`. Depending on the computational power of different GPUs, the generation process may take between 10 to 30 hours on an 8-GPU machine. To speed up the generation, the script can be modified to distribute the workload across multiple machines.
bash eval_mjhq.sh

```



# VQA benchmarks evaluation
For model evaluation, please follow this [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for data preparation.

```bash
cd VQA_Eval

# TextVQA 44.31%
bash textvqa.sh

# GQA 57.48%
bash gqa.sh

# POPE 
bash pope.sh

#VQAv2
bash vqav2.sh
 

```


# Language Tasks
Liquid can be directly utilized as a language model, and we employ [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as the evaluation tool to report results, for instance:

```bash
# install lm_eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .

#single-GPU

lm_eval --model hf \
    --model_args pretrained=/path/to/Liquid_V1_7B,dtype="float" \
    --tasks  hellaswag,winogrande,arc_easy,arc_challenge,boolq,mmlu \
    --device cuda:0 \
    --batch_size 8


# multi-GPU

accelerate launch  --main_process_port 9999 -m lm_eval  --model hf \
    --model_argspretrained=/path/to/Liquid_V1_7B,dtype="float" \
    --tasks  hellaswag,winogrande,arc_easy,arc_challenge,boolq,mmlu \
    --batch_size 8 
```