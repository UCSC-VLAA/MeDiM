import torch
import numpy as np
import argparse
import time
import PIL
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import os
from tqdm import tqdm
from chameleon.inference.image_tokenizer import ImageTokenizer
from VQA_Eval.conversation import  conv_templates
from threading import Thread
from T2I_Eval.genaibench_generation import sample

def main(args):
    
    temperature = args.temperature
    guidance_scale = args.cfg
    top_K = args.TopK
    top_P = args.TopP
    image_save_pth = args.save_path
    if not os.path.exists(image_save_pth):
        os.makedirs(image_save_pth)

    
    assert temperature <= 1.0
    assert top_K <= 8192
    assert top_P <= 1.0

    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side='left')
    ori_vocabe_size = len(tokenizer)

    vqllm = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
        )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')

    vqgan_cfg_path = "chameleon/vqgan.yaml"
    vqgan_ckpt_path = "chameleon/vqgan.ckpt"
    image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0",)

    text_inputs = [args.prompt]*4  # generate 4 samples once
    uncondition_text_inputs = ['<unconditional><boi>']*len(text_inputs)
    for i in range(len(text_inputs)):
        text_inputs[i] = text_inputs[i]+' Generate an image based on this description.<boi>'

    if guidance_scale>1:
        model_inputs = tokenizer(text_inputs+uncondition_text_inputs, return_tensors="pt",padding=True).to("cuda:0")
    else:
        model_inputs = tokenizer(text_inputs, return_tensors="pt",padding=True).to("cuda:0")
    with torch.no_grad():
        sampling_kwargs={'temperature': temperature, 'top_k': top_K, 'top_p': top_P, 'sample_logits': True}
        input_ids = model_inputs['input_ids']
        cur_len = input_ids.shape[1]
        model_kwargs = {'attention_mask':model_inputs['attention_mask']  , 'use_cache': True}
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        pred_tokens = []
        for i in tqdm(range(1024)):
            model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)

            if i > 0 and guidance_scale>1:
                outputs = vqllm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            else:
                outputs = vqllm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )

            next_token_logits = outputs.logits[:, -1:, :]
            
            if guidance_scale>1:
                cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0) 
                cfg_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_scale
                half_next_token, _ = sample(cfg_logits, **sampling_kwargs)
                pred_tokens.append(half_next_token)
                next_token = torch.cat([half_next_token,half_next_token])


            else:
                next_token, next_prob = sample(next_token_logits, **sampling_kwargs)
                pred_tokens.append(next_token)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            model_kwargs = vqllm._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=vqllm.config.is_encoder_decoder,
            )

        del sampling_kwargs
        del model_inputs
        del outputs
        image_vq_id = torch.cat(pred_tokens,dim=1)-ori_vocabe_size
        image_vq_id = torch.clamp(image_vq_id, min=0, max=8191)
        
        generated_image_list = []
        for index, generate_id in enumerate(image_vq_id):
            rec_img = image_tokenizer.pil_from_img_toks(generate_id)
            generated_image_list.append(rec_img)
            rec_img.save('{}/sample_{}.jpg'.format(image_save_pth,str(index)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str,default='Junfeng5/Liquid_V1_7B', help='model path, default to huggingface repo id')
    parser.add_argument('--save_path', type=str,default='samples/t2i', help='save path')
    parser.add_argument('--prompt', type=str, required=True, help='input text prompt')
    parser.add_argument('--load_8bit',  action='store_true', default=False, help='use 8bit to save memory')
    parser.add_argument('--cfg', type=float,default=7.0, help='Classifier-Free Guidance scale')
    parser.add_argument('--TopP', type=float,default=0.96, help='Top P, max=1.0')
    parser.add_argument('--TopK', type=int,default=4096, help='Top K, max=8192')
    parser.add_argument('--temperature', type=float,default=0.99, help='sampling temperature, max=1.0')


    args = parser.parse_args()
    main(args)


