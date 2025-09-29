import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from PIL import ImageFile
from PIL import Image
import math
import sys
from torchvision import transforms
import PIL
sys.path.append('../')
from chameleon.inference.image_tokenizer import ImageTokenizer
import  numpy as np

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

ImageFile.LOAD_TRUNCATED_IMAGES = False
torch.set_grad_enabled(False)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        # import pdb;pdb.set_trace()
        
        # if self.model_config.mm_use_im_start_end:
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = prompt.replace('<image>','<boi><image><eoi>')
        # import pdb;pdb.set_trace()
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        pad_image = expand2square(image, (122, 116, 104) )
        input_image = pad_image.resize((512,512), PIL.Image.LANCZOS)


        with torch.no_grad():
            vq_code =  self.image_processor.img_tokens_from_pil(input_image) 
            vqcode = vq_code.cpu() 
            vqcode = vqcode+ len(self.tokenizer)
 
        text_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        eoi =  torch.tensor([8])
        boi =  torch.tensor([7])
        image_token_indices = [-1] + torch.where(text_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                input_vqcodes = torch.cat( [boi,vqcode,eoi],dim=0 )
                cur_input_ids.append( input_vqcodes )
        input_ids = torch.cat(cur_input_ids, dim=0)

        return input_ids,os.path.join(self.image_folder, image_file) #, image_tensor, image_tensor_aux
    
    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=0):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, load_8bit=args.load_8bit)

    
    kwargs = {}
    kwargs = {"device_map": "cuda"}
    if args.load_8bit:
        kwargs['load_in_8bit'] = True
    else:
        kwargs['torch_dtype'] = torch.float16

    if args.use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    # import pdb;pdb.set_trace()

        
    vqgan_cfg_path = "../chameleon/vqgan.yaml"
    vqgan_ckpt_path = "../chameleon/vqgan.ckpt"
    image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda",)

 
    image_processor = image_tokenizer
    # import pdb;pdb.set_trace()

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids ,imagepath), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]
        
        input_ids = input_ids.to(device=model.device, non_blocking=True)
        if hasattr(model, "update_prompt"):
            model.update_prompt([[cur_prompt]])
        with torch.inference_mode():
            
            inputs_embeds = model.model.embed_tokens(input_ids)
            output_ids = model.generate(
                inputs_embeds=inputs_embeds,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=tokenizer.pad_token_id,  # Pad token
                use_cache=False)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # print('imagepath\n',imagepath)
        # print('question\n',cur_prompt)
        # print('answer:\n', outputs)
        # import pdb;pdb.set_trace()  # cur_prompt,outputs
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--use_flash_attn', type=bool, default=True)
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
