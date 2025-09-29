import transformers
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, GemmaForCausalLM
from torch import nn
import torch
import  json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import argparse
from torch.nn import functional as F
import sys
sys.path.append('../')

from chameleon.inference.image_tokenizer import ImageTokenizer
import  numpy as np

### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
    ):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    logits[:,:256000]=filter_value # sampling on VQVAE vocabulary for image generation
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs



def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_path', type=str, default='/path/to/Liquid_models//Liquid_V1_7B/')
    parser.add_argument('--save_path', type=str, default='mjhq30k_results')
    parser.add_argument('--load_8bit', type=bool, default=False)
    parser.add_argument('--chunk_idx', type=int, default=0)
    parser.add_argument('--num_chunks', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=7.0)
    parser.add_argument('--tau', type=float, default=0.99)
    parser.add_argument('--topk', type=int, default=4096)
    parser.add_argument('--topp', type=float, default=0.96)
    return parser

def split_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def main(args):
    LLM_pth = args.model_path
    text_set_id = args.chunk_idx
    cfg_scale = args.cfg_scale
    use_catch = True
    tau = args.tau
    topk = args.topk
    topp = args.topp
    num_chunks=args.num_chunks
    batch_size = args.batch_size
    image_save_pth = '{}/CFG{}_topk{}_topp{}_tau_{}'.format(args.save_path,str(cfg_scale), str(topk),str(topp),str(tau))
    tokenizer = AutoTokenizer.from_pretrained(LLM_pth,padding_side='left')
    vqllm = AutoModelForCausalLM.from_pretrained(
        LLM_pth,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        load_in_8bit=args.load_8bit,
        )
    if not args.load_8bit:
        vqllm = vqllm.to('cuda')

    ori_vocabe_size = len(tokenizer)

    all_datas = json.load(open('MJHQ-30K/meta_data.json','rb'))
    all_prompts = []
    prompts_lens = []
    for k,v in all_datas.items():
        v.update({'name':k})
        prompts_lens.append(len(v['prompt']))
        all_prompts.append(v)
    
    chunked_filenames = np.array_split(all_prompts, num_chunks)
    subset = chunked_filenames[text_set_id].tolist()
    chunk_inputs = split_list(subset, batch_size)

    for chunk in tqdm(chunk_inputs):

        text_inputs = [v['prompt'] for v in chunk]
        # text_inputs = [t if len(t)<600 else t[:600] for t in text_inputs ] # for Limited VRAM GPUS like 4090 to avoid OOM
        uncondition_text_inputs = ['<unconditional><boi>']*len(text_inputs)
        for i in range(len(text_inputs)):
            text_inputs[i] = text_inputs[i]+' Generate an image based on this description.<boi>'
        save_list = []
        if cfg_scale>1:
            model_inputs = tokenizer(text_inputs+uncondition_text_inputs, return_tensors="pt",padding=True).to('cuda')
        else:
            model_inputs = tokenizer(text_inputs, return_tensors="pt",padding=True).to('cuda')
        with torch.no_grad():
            sampling_kwargs={'temperature': tau, 'top_k': topk, 'top_p': topp, 'sample_logits': True}
            input_ids = model_inputs['input_ids']
            cur_len = input_ids.shape[1]
            model_kwargs = {'attention_mask':model_inputs['attention_mask']  , 'use_cache': True}
            model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

            pred_tokens = []
            for i in range(1024):
                model_inputs = vqllm.prepare_inputs_for_generation(input_ids, **model_kwargs)
                if i > 0 and cfg_scale>1:
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
                
                if cfg_scale>1:
                    cond_logits, uncond_logits = torch.split(next_token_logits, len(next_token_logits) // 2, dim=0) 
                    cfg_logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
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
            image_list = []
            for index, generate_id in enumerate(image_vq_id):
                image_list.append(generate_id.tolist())
            save_list.append(image_list)

        torch.cuda.empty_cache()

        vqgan_cfg_path = "../chameleon/vqgan.yaml"
        vqgan_ckpt_path = "../chameleon/vqgan.ckpt"
        image_tokenizer = ImageTokenizer(  cfg_path=vqgan_cfg_path, ckpt_path=vqgan_ckpt_path, device="cuda",)

        for datainfo, vq_code in zip(chunk, save_list[0]):
            name = datainfo['name']
            category = datainfo['category']
            sub_path = os.path.join(image_save_pth, category)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

            latents = torch.tensor(vq_code).to('cuda')
            rec_img = image_tokenizer.pil_from_img_toks(latents)
            rec_img.save('{}/{}.png'.format(sub_path,name))
        del image_tokenizer
        torch.cuda.empty_cache()
        print(text_set_id,' is done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser('image path check script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)