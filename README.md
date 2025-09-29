# MedDiM

## Getting Started

Step1. To install the dependencies, run:
```bash
# create new anaconda env
conda create -n MedDiM python=3.10
conda activate MedDiM 

# install packages
pip install -r requirements.txt
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Step2. Prepare mimic-cxr dataset and PathGen dataset in Huggingface, run:
```bash
# downloading mimic-cxr from Huggingface
huggingface-cli login
huggingface-cli snapshot download MLforHealthcare/mimic-cxr --local-dir ./dataset/mimic-cxr --local-dir-use-symlinks False
```

Step3. Prepare pretrain checkpoint, run:
```bash
# download VQVAE config and weight
cd ./models
wget -P chameleon/ https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.ckpt 
wget -P chameleon/ https://huggingface.co/spaces/Junfeng5/Liquid_demo/resolve/main/chameleon/vqgan.yaml

# download Liquid 7B
huggingface-cli login
huggingface-cli snapshot download Junfeng5/Liquid_V1_7B --local-dir ./models/Liquid_V1_7B --local-dir-use-symlinks False

# download Llama-2-7b-hf
huggingface-cli login
huggingface-cli snapshot download NousResearch/Llama-2-7b-hf --local-dir ./models/Llama-2-7b-hf --local-dir-use-symlinks False
```

Step4. Fixing `num_hidden_layers: 10` of `config.json` in ./models/Liquid_V1_7B.

Step5. MedUnidisc training, run:
```bash
# training
accelerate launch  --num_processes 8 --multi_gpu --main_process_port=$RANDOM main.py +experiments='[large_scale_train]' debug=true loader.batch_size=1 data.data_dir_train=./dataset/mimic-cxr/data data.data_dir_val=./dataset/mimic-cxr/data model.vqgan_config=./models/chameleon/vqgan.yaml model.vqgan_ckpt=./models/vqgan_ckpt model.llama_ckpt=./models/Llama-2-7b-hf model.liquid_ckpt=./models/Liquid_V1_7B
```

Step6. Find the latest ckpt path, run:
```bash
# find ckpt path
python find_latest_ckpt.py ./medunidisc/outputs/outputs/debug
```

Step7. Resume MedUnidisc training, run:
```bash
# resume
accelerate launch  --num_processes 8 --multi_gpu --main_process_port=$RANDOM main.py +experiments='[large_scale_train]' debug=true loader.batch_size=1 data.data_dir_train=./dataset/mimic-cxr/data data.data_dir_val=./dataset/mimic-cxr/data model.vqgan_config=./models/chameleon/vqgan.yaml model.vqgan_ckpt=./models/vqgan_ckpt model.llama_ckpt=./models/Llama-2-7b-hf model.liquid_ckpt=./models/Liquid_V1_7B
```

