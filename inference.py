import os
import sys
from contextlib import ExitStack
from pathlib import Path

from constants import CONFIG_PATH, LIB_DIR
sys.path.append(str(LIB_DIR / "hydra_submitit_launcher"))
import hydra
import builtins
import random
import re
import signal
import traceback
from copy import deepcopy
from datetime import datetime
from evaluation.chameleon.inference.image_tokenizer import ImageTokenizer
from transformers import AutoTokenizer
import numpy as np
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict, read_write
from safetensors.torch import load_file, save_file
from models.datasets.all_dataset import TrainDataset, ValDataset
import dataloader
from model_inf import Diffusion
import utils
from decoupled_utils import (check_gpu_memory_usage, is_main_process, get_rank,
                             rprint, rank_zero_fn, print_params, is_torch_cuda_available,
                             set_timing_builtins, try_except)
from utils import (ErrorHandler, _print_config, convert_state_dict_keys, set_torch_defaults, set_omega_conf_resolvers)


set_omega_conf_resolvers()

def _load_from_checkpoint(config, text_tokenizer, image_tokenizer):
    OmegaConf.resolve(config)
    if "hf" in config.backbone:
        return Diffusion(config=config, text_tokenizer=text_tokenizer, image_tokenizer=image_tokenizer).to("cuda")

    return Diffusion.load_from_checkpoint(config.eval.checkpoint_path, tokenizer=text_tokenizer, config=config)

@rank_zero_fn
def _print_batch(train_ds, valid_ds, txt_tokenizer, k=256):
    for dl_type, dl in [("train", train_ds), ("valid", valid_ds)]:
        print(f"Printing {dl_type} dataloader batch.")
        batch = next(iter(dl))
        print("Batch input_ids.shape", batch["text"].shape, batch["image"].shape)
        first = batch["text"][0]
        print(f"text {k} tokens:", txt_tokenizer.decode(first.long().tolist()).replace('<unk>', ''))
        

def generate_samples(config, text_tokenizer, image_tokenizer):
    print("Generating samples.")
    model = _load_from_checkpoint(config=config, text_tokenizer=text_tokenizer, image_tokenizer=image_tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        print("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides
    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length, num_strides=num_strides, dt=1 / config.sampling.steps
            )
            text_samples = intermediate_samples[-1]
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            model.compute_generative_perplexity(text_samples)
    
    print("Text samples:", text_samples)
    if not config.sampling.semi_ar:
        print("Generative perplexity:", model.gen_ppl_metric.compute())
    return text_samples


def update_config_before_resolution(config):
    import torch
    if hasattr(config, "training"):
        print(f"'training' has been refactored to 'trainer'. Please update the config.")
        
    with open_dict(config):
        config.output_dir = os.getcwd()
        config.logging_dir = os.getcwd()
        if config.model.use_kv_cache is False and config.mode == "eval" and config.loader.eval_batch_size > 1:
            config.loader.eval_batch_size = 1

def save_config_to_ckpt(config, output_dir, model):
    with try_except(write_error_to_file=True, clear_cuda_cache=True):
        with read_write(config):
            with open_dict(config):
                config.state.ckpt_step = model.global_step
                config.state.num_evals = model.num_evals

        OmegaConf.save(config=config, f=Path(output_dir) / "config.yaml")
        print(f"Saved global step {model.global_step}")

def run(config, text_tokenizer):
    import torch
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    from accelerate.utils import GradientAccumulationPlugin, ProjectConfiguration

    update_config_before_resolution(config)
    OmegaConf.resolve(config)
    sync_timing = (config.trainer.nvtx_profile and getattr(config.trainer, "sync_nvtx_timing", True)) or getattr(config.trainer, "sync_timing", False)
    set_timing_builtins(enable=config.trainer.nvtx_profile, sync=sync_timing)
    with open_dict(config):
        config.trainer = OmegaConf.merge(config.trainer, dict(mixed_precision=config.trainer.precision, log_with=None, log_gradients=None))


    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=config.logging_dir,
    )

    accelerate_kwargs = dict()

    if config.trainer.mixed_precision == "bf16":
        rprint(f"No BF16 GPU found, falling back to FP16")
        config.trainer.mixed_precision = "fp16"

    if config.trainer.mixed_precision == "fp32":
        config.trainer.mixed_precision = "no"

    rprint(f"Mixed precision: {config.trainer.mixed_precision}")


    accelerator = Accelerator(
        mixed_precision=config.trainer.mixed_precision,
        log_with=config.trainer.log_with,
        project_config=accelerator_project_config,
        **accelerate_kwargs,
    )

    num_processes = AcceleratorState().num_processes

    if not config.trainer.disable_adjust_num_warmup_steps:
        rprint(f"Original num_warmup_steps was: {config.lr_scheduler.num_warmup_steps}")
        config.lr_scheduler.num_warmup_steps = config.lr_scheduler.num_warmup_steps * num_processes
        rprint(f"Setting num_warmup_steps to: {config.lr_scheduler.num_warmup_steps}")

        if hasattr(config.lr_scheduler, "num_training_steps"):
            rprint(f"Original num_training_steps was: {config.lr_scheduler.num_training_steps}")
            config.lr_scheduler.num_training_steps = config.lr_scheduler.num_training_steps * num_processes
            rprint(f"Setting num_training_steps to: {config.lr_scheduler.num_training_steps}")

    compute_dtyle = torch.float32
    if accelerator.mixed_precision == "fp16":
        compute_dtyle = torch.float16
    elif accelerator.mixed_precision == "bf16":
        compute_dtyle = torch.bfloat16

    if compute_dtyle != torch.bfloat16:
        print(f"WARNING!!!! Compute dtype is: {compute_dtyle}")
    else:
        print(f"Compute dtype is: {compute_dtyle}")

    with open_dict(config):
        config.trainer.devices = accelerator.num_processes
        config.trainer.dtype = str(compute_dtyle)

    OmegaConf.set_readonly(config, True)
    #print(f'dataset dir:{config.data.data_dir_train}')
    train_dataset, val_dataset = TrainDataset(config.data.data_mimic_dir_train, config.data.data_path_dir_train, txt_tokenizer=text_tokenizer), \
                                 ValDataset(config.data.data_mimic_dir_val, config.data.data_path_dir_val, txt_tokenizer=text_tokenizer)
    image_tokenizer = ImageTokenizer(cfg_path=config.model.vqgan_config, ckpt_path=config.model.vqgan_ckpt, device=accelerator.device)
    from torch.utils.data import DataLoader
    print(f'train dataset:{len(train_dataset)}, val dataset:{len(val_dataset)}')
    train_ds, valid_ds = DataLoader(train_dataset, batch_size=1, shuffle=True), DataLoader(val_dataset, batch_size=1)
    model = Diffusion(config=config, text_tokenizer=text_tokenizer, image_tokenizer=image_tokenizer, device=accelerator.device)

    if accelerator.is_main_process:
        print_params(model.backbone)

    _print_batch(train_ds, valid_ds, text_tokenizer)

    def save_model_hook(models, weights, output_dir):
        nonlocal model, accelerator, train_ds

        if is_main_process():
            with try_except(write_error_to_file=True):
                if getattr(model, "ema", None) is not None:
                    torch.save(accelerator.unwrap_model(model).ema.state_dict(), os.path.join(output_dir,'ckpt','checkpoint.ckpt'))
                    print(f"Saved EMA to {os.path.join(output_dir,'ckpt','checkpoint.ckpt')}")
            #save_config_to_ckpt(config, output_dir, model)

    initial_global_step = None
    def load_model_hook(models, input_dir):
        nonlocal initial_global_step, model, train_ds
        config_path = os.path.join(input_dir, "config.yaml")
        ckpt_config = OmegaConf.load(config_path)
        initial_global_step = ckpt_config.state.ckpt_step
        model.global_step = initial_global_step
        try:
            if hasattr(config.state, "num_evals"):
                model.num_evals = config.state.num_evals
        except Exception as e:
            print(f"Error loading model: {e}")
        print(f"Loaded global step {initial_global_step}")

        state_dict = None

        if getattr(model, "ema", None) is not None:
            print(f"Loading EMA from {os.path.join(input_dir,'ckpt','checkpoint.ckpt')}")
            model.ema.load_state_dict(torch.load(os.path.join(input_dir,'ckpt','checkpoint.ckpt'), map_location='cpu'))
        else:
            rprint(f"No EMA found, initializing EMA with state_dict")
            if state_dict is None:
                state_dict = load_file(os.path.join(input_dir, "model.safetensors"))

            # We likely don't need the unwrap, but just to be safe
            accelerator.unwrap_model(models[0]).load_state_dict(state_dict)
            
            from models.ema import EMAModel
            model.ema = EMAModel(accelerator.unwrap_model(models[0]).parameters(), decay=config.trainer.ema)


    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    model.init_dataloader(train_ds, valid_ds)
    model.set_accelerator(accelerator, None)

    if initial_global_step is not None:
        # The load_hooks are before accelerate does it's loading and it overwrites model.global_step if we set it there
        model.global_step = initial_global_step
        print(f"Set global step to {initial_global_step}")

    print(f"output_dir: {config.output_dir}")

    if config.trainer.load_from_state_dict is not None:
        model.global_step = int(config.trainer.load_from_state_dict.split('_')[-1])

    model.to(accelerator.device)
    
    model.validate(config.eval.task)

    accelerator.end_training()



@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(config):

    """Main entry point for trainer."""
    import torch  # Causes issue pickling if imported by default.
    from unidisc.utils.logging_utils import set_logger

    if config.seed is not None:
        if config.mode == 'eval':
            config.seed = config.seed + 1000 * int(get_rank())
        else:
            config.seed = config.seed + int(get_rank())
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if is_torch_cuda_available():
            # TODO: Is seed all desired? Does it set the same one on all GPUs even in multi-process?
            torch.cuda.manual_seed_all(config.seed)
    else:
        rprint("No seed provided")

    _print_config(config, resolve=True, save_cfg=True)

    text_tokenizer = AutoTokenizer.from_pretrained(config.model.llama_ckpt, padding_side='right')
    special_tokens_dict = {
    "additional_special_tokens": ["<boi>", "<eoi>", "<eos>"]
    }
    text_tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Mode: {config.mode}")
    if config.mode == "sample_eval":
        generate_samples(config, text_tokenizer)
    else:
        run(config, text_tokenizer)


if __name__ == "__main__":
    main()
