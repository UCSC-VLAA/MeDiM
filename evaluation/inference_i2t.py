import torch
import argparse
import PIL
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from chameleon.inference.image_tokenizer import ImageTokenizer
from VQA_Eval.conversation import  conv_templates
from threading import Thread



IMAGE_TOKEN_INDEX=-200

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
    vqgan_cfg_path = "chameleon/vqgan.yaml"
    vqgan_ckpt_path = "chameleon/vqgan.ckpt"
    image_tokenizer = ImageTokenizer(cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda:0",)

    qs = args.prompt
    qs = '<boi><image><eoi>' + '\n' + qs
    conv = conv_templates['gemma'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(prompt)
    image = Image.open(args.image_path).convert('RGB')
    pad_image = expand2square(image, (122, 116, 104) )
    input_image = pad_image.resize((512,512), PIL.Image.LANCZOS)
    with torch.no_grad():
        vq_code =  image_tokenizer.img_tokens_from_pil(input_image) 
        vqcode = vq_code.cpu() 
        vqcode = vqcode+ len(tokenizer)

        text_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        num_images = (text_ids == IMAGE_TOKEN_INDEX).sum()
        image_token_indices = [-1] + torch.where(text_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [text_ids.shape[0]]
        cur_input_ids = []
        for i in range(num_images + 1):
            cur_input_ids.append(text_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            if i < num_images:
                cur_input_ids.append( vqcode )
        input_ids = torch.cat(cur_input_ids, dim=0)
        # input_embeddings = vqllm.embed_tokens(input_ids)
        inputs =  {
            "input_ids":input_ids.unsqueeze(0).to("cuda:0"),
            "max_new_tokens":1024,
            "bos_token_id":tokenizer.bos_token_id,  # Begin of sequence token
            "eos_token_id":tokenizer.eos_token_id,  # End of sequence token
            "pad_token_id":tokenizer.pad_token_id,  # Pad token
            }
        streamer = TextIteratorStreamer(tokenizer, **{"skip_special_tokens": True, "skip_prompt": True})

        # Run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way.
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1024)
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
    parser.add_argument('--image_path', type=str, required=True, help='input image path')
    parser.add_argument('--load_8bit',  action='store_true', default=False, help='use 8bit to save memory')

    args = parser.parse_args()
    main(args)


