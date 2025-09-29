import torch
import argparse
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from transformers.models.
import os
from VQA_Eval.conversation import  conv_templates
from threading import Thread


def main(args):
    
    model_id = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_id,padding_side='left')
    vqllm = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
        )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')
    # vqgan_cfg_path = "chameleon/vqgan.yaml"
    # vqgan_ckpt_path = "chameleon/vqgan.ckpt"
    # image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0",)

    conv = conv_templates['gemma'].copy()
    conv.append_message(conv.roles[0], args.prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(prompt)
    with torch.no_grad():
        inputs = tokenizer([prompt], return_tensors="pt").to('cuda')

        # outputs = vqllm.generate(**inputs, do_sample=True, max_new_tokens=256)
        # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

        # use generate() to get results or use TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, **{"skip_special_tokens": True, "skip_prompt": True})
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=200)
        thread = Thread(target=vqllm.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
        print(generated_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str,default='Junfeng5/Liquid_V1_7B', help='model path, default to huggingface repo id')
    parser.add_argument('--prompt', type=str, required=True, help='input text prompt')
    parser.add_argument('--load_8bit',  action='store_true', default=False, help='use 8bit to save memory')

    args = parser.parse_args()
    main(args)


