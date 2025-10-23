import math
import random
import types
import time
from collections import defaultdict
from contextlib import nullcontext
from functools import cached_property, partial
from contextlib import ExitStack

from numpy import mask_indices
from unidisc.utils.tensor_utils import get_contiguous_blocks, get_contiguous_blocks_per_sample, get_interleaved_indices
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.utils import gather, gather_object
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor, nn
from tqdm.auto import tqdm

import model_eval_inf
import model_setup
import model_utils
import utils
from decoupled_utils import (Profiler, barrier, dprint, get_rank, get_world_size, gprint,
                             is_local_main_process, is_main_process,
                             is_torch_cuda_available, is_torch_xla_available,
                             print_memory, rprint, save_memory_profile,
                             synchronize_device, try_except, use_dist)
from unidisc.tokenizers.image_tokenizers import (decode_latents, get_image_batch,
                                              get_vae, vae_encode_image)
from unidisc.utils.cuda_utils import sync_times
from unidisc.utils.xla_utils import shard_output
from model_utils import (Loss, ddprint, ema_update, empty_device_cache, get_chameleon_txt_indices, get_interleaved_block_mask, log,
                         replace_nan_dict, update_histogram, update_logs, get_block_mask)
from unidisc.utils.trainer_utils import TrainingState, incremental_dict_update, linear_warmup


class Diffusion:
    def __init__(self, config, text_tokenizer, image_tokenizer, device, disable_init=False):
        super().__init__()
        setup_methods = [
            'init', 'to', 'get_params', 'configure_optimizers',
            '_validate_configuration', 'register_signal_handler', 'on_train_start',
            'optimizer_step', 'init_dataloader', 'set_accelerator', 'set_callbacks', 
            'on_train_step_end', 'init_optimizer_lr_scheduler', 'after_backward', 'checkpoint', 
            'shortcut_return', 'reset_validation_metrics', 'unwrap_model'
        ]
        for method_name in setup_methods:
            setattr(self, method_name, types.MethodType(getattr(model_setup, method_name), self))

        utils_methods = [
            'get_coord_plot', '_score_entropy', 'sample_subs_guidance',
            'restore_model_and_semi_ar_sample', '_reconstruction_loss',
            'restore_model_and_sample', 'get_score', '_staggered_score',
            '_analytic_update', '_denoiser_update', '_transp_transition',
            'eval_retokenize', 'compute_generative_perplexity', '_d3pm_loss',
            '_d3pm_parameterization', '_sedd_parameterization',
            'get_base_shapes_for_mup', 'update_histogram', '_maybe_sub_sample',
             'viz_images_from_dataloader', 'compute_cider'
        ]
        for method_name in utils_methods:
            setattr(self, method_name, types.MethodType(getattr(model_utils, method_name), self))

        eval_methods = [
            'get_every_n_evals', 'on_validation_epoch_start', 'sample',
            'predict_step', 'validation_step', 'on_validation_epoch_end',
            'on_validation_epoch_cleanup', '_sample_prior', '_ddpm_forward',
            '_ddpm_update', '_ddpm_caching_update', '_sample', '_ar_sampler',
            'decode_batch', 'sample_transfusion', 'sample_continuous_image',
            'decode_sampling', '_ddpm_update_finetune_controlled_tweedie', 
            'sample_masking', 'log_flops', "visualize_samples", "_maskgit_update", 
            "_first_hitting_update", "update_inline_fid", "compute_inline_fid",
            "update_clean_fid", "compute_clean_fid_eval", "sample_for_fid",
            "compute_clip_score", "mauve_store_references", "zero_shot_eval_step",
            "zero_shot_eval_epoch_end", "get_cfg_weight", "cleanup_fid_output",
            "calculate_chameleon_perplexity", "get_anole_data",
            "update_img_to_txt_mauve_clip", "compute_mauve_entropy",
            "get_top_k", "compute_entropy", "get_mauve_score", "get_valid_seq", "gather_tokens",
            "count_valid_tokens", "compute_val_metrics_standalone", "_maskgit_nucleus_update",
            "get_img_text_saturation_batch", "handle_interleaved_decode", "get_interleaved_image",
            "auto_enhance", "get_clip_score", "get_dfn_score", "get_hpsv2_score", "get_model_likelihood_score",
            "get_laion_aesthetic_score", "get_rewards", "get_chameleon_score", "clear_reward_models",
            "get_text_likelihood_score", "get_text_reward_model_score", "save_image_text_pair"
        ]
        for method_name in eval_methods:
            setattr(self, method_name, types.MethodType(getattr(model_eval_inf, method_name), self))

        if disable_init:
            pass
        else:
            model_setup.init(self, config, device)

        self.text_tokenizer = text_tokenizer
        self.image_tokenizer = image_tokenizer
        #self.backbone = backbone

    def on_train_resume(self):
        empty_device_cache()
        if self.ema is not None and not self.config.trainer.use_custom_ema:
            self.ema.restore(self.get_params(), raise_error_if_already_restored=False)
        self.backbone.train()

    def update_batch(self, batch):

        image = batch['image']
        caption = batch['text'].to(torch.int64)
        sizes = batch['size']
        #attn_txt_mask = batch['mask']
        #batch['img_mask'] = torch.ones_like(image)
        batch_size = image.size()[0]

        txt_tokens = caption
        #print(f'txt:{txt_tokens}')
        with torch.no_grad():
            _, _, [_, _, vq_code] = self.image_tokenizer._vq_model.encode(image)
            vq_code = vq_code.view(image.shape[0], -1)
            vq_code = vq_code + len(self.text_tokenizer) + 1
            img_tokens = vq_code.to(torch.int64)

        batch['img_mask'] = torch.ones_like(img_tokens)

        cond_boi, cond_eoi, cond_eos = ['<boi>'], ['<eoi>'], ['<eos>']
        boi_ids = self.text_tokenizer(
            cond_boi,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
        ).input_ids[0][1].unsqueeze(dim=0).repeat(batch_size, 1).to(torch.int64)

        eoi_ids = self.text_tokenizer(
            cond_eoi,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
        ).input_ids[0][1].unsqueeze(dim=0).repeat(batch_size, 1).to(torch.int64)

        eos_ids = self.text_tokenizer(
            cond_eos,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
        ).input_ids[0][1].unsqueeze(dim=0).repeat(batch_size, 1).to(torch.int64)

        img_cond = ['Image Size: Width is {} Height is {}.'.format(sizes[0], sizes[1])]
        cond_ids = self.text_tokenizer(
            img_cond,
            return_tensors="pt",
            padding="longest",
            max_length=256,
            truncation=True,
        ).input_ids[0].unsqueeze(dim=0).repeat(batch_size, 1).to(torch.int64)
        
        #print('input:',txt_tokens.shape, img_tokens.shape)
        batch["input_ids"] = torch.cat([txt_tokens, img_tokens], dim=-1).to(self.device)
        batch["boi"], batch["eoi"], batch["eos"], batch["cond"] = boi_ids.to(self.device), eoi_ids.to(self.device),  \
                                                                  eos_ids.to(self.device), cond_ids.to(self.device)

        return batch

    def training_step(self, batch, batch_idx, epoch):
        batch = self.update_batch(batch)
        return self.compute_loss(batch, prefix="train", batch_idx=batch_idx, epoch=epoch)

    def q_xt(self, x, move_chance, allow_move_mask=None, return_ignore_batch_mask_for_metrics=False, mask_image_square=False, mask_text_region=False, batch=None):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        ignore_batch_mask_for_metrics = None
        should_mask_txt, should_mask_img = None, None
        joint_ar_nar_mask = None
        if allow_move_mask is not None:
            move_indices = move_indices & allow_move_mask
        xt = torch.where(move_indices, self.mask_index, x)
        if random.random() < 0.1:
            prompt_length = batch['prompt_idx'].squeeze().item()
            xt[:, :prompt_length] = x[:, :prompt_length]
        if return_ignore_batch_mask_for_metrics:
            return xt, ignore_batch_mask_for_metrics, joint_ar_nar_mask, should_mask_txt, should_mask_img, move_indices
        else:
            return xt

    def _sample_t(self, n, device, epoch):

        _eps_t = torch.rand(n, device=device)
        offset = torch.arange(n, device=device) / n
        _eps_t = (_eps_t / n + offset) % 1

        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps

        return t.to(torch.float32)

    def _subs_parameterization(self, logits, xt, batch=None, modality=None, **kwargs):
        # log prob at the mask index = - infinity
        if not self.allow_slicing:
            logits = logits.clone()

        logits[..., self.mask_index] += self.neg_infinity
        if getattr(self.config.model, "force_argmax_valid_indices", False):
            logits[..., self.static_txt_sl, self.text_vocab_size:] = self.neg_infinity
            logits[..., self.static_img_sl, :self.text_vocab_size] = self.neg_infinity

        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        if self.parameterization != "ar" and xt is not None:

            unmasked_indices = xt != self.mask_index
            if not self.allow_slicing:
                logits = torch.where(unmasked_indices.unsqueeze(-1), torch.full_like(logits, self.neg_infinity),  logits)
                logits = torch.where(
                    unmasked_indices.unsqueeze(-1) & (torch.arange(logits.size(-1)).to(logits.device) == xt.unsqueeze(-1)),  
                    torch.zeros_like(logits),
                    logits
                )
            else:
                logits[unmasked_indices] = self.neg_infinity
                logits[unmasked_indices, xt[unmasked_indices]] = 0

        return logits

    def _process_sigma(self, sigma):
        if sigma.ndim > 1 and not self.config.trainer.image_mode == "continuous":
            sigma = sigma.squeeze(-1)
            assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(
        self,
        x,
        sigma,
        batch=None,
        return_logits=False,
        block_mask=None,
        update_cache_slice=None,
        **kwargs,
    ):
        """Returns log score."""
        sigma = self._process_sigma(sigma)
        txt_token, img_token = x[:,:256], x[:,256:]
        bsz = x.shape[0]
        attn_txt_mask, attn_img_mask = batch['mask'], batch['img_mask']
    
        c_bsz = batch['boi'].shape[0]
    
        if bsz != c_bsz:
            input_ids = torch.cat([batch['boi'].repeat(bsz,1), img_token,
                                   batch['eoi'].repeat(bsz,1), batch['cond'].repeat(bsz,1),
                                   txt_token, batch['eos'].repeat(bsz,1)], dim=-1)
            attn_one_token = torch.ones_like(batch['boi'].repeat(bsz,1))
            attn_one_tokens = torch.ones_like(batch['cond'].repeat(bsz,1))
            attn_img_mask = attn_img_mask.repeat(bsz,1)
        else:
            input_ids = torch.cat(
                [batch['boi'], img_token, batch['eoi'], batch['cond'],
                 txt_token, batch['eos']], dim=-1)
            attn_one_token = torch.ones_like(batch['boi'])
            attn_one_tokens = torch.ones_like(batch['cond'])

        attention_mask = torch.cat([attn_one_token, attn_img_mask, attn_one_token, attn_one_tokens, txt_token, attn_one_token], dim=-1)

        # if self.config.trainer.image_mode == "continuous": assert "modality" in kwargs
        should_autocast = (((self.config.trainer.disable_forward_autocast_during_eval and self.backbone.training) is False) and (self.dtype != torch.float32))
        with ExitStack() as stack:
            if should_autocast:
                stack.enter_context(torch.autocast(device_type=self.device.type, dtype=self.dtype))
            
            logits = self.backbone(input_ids=input_ids, attention_mask=attention_mask, time=sigma).logits

        xt = x
        if self.parameterization == "subs":
            if return_logits:
                return logits
            model_output = self._subs_parameterization(logits, xt=xt, batch=batch, **kwargs)
            return model_output

        return logits

    def compute_loss(self, batch, prefix, batch_idx=-1, epoch=-1):

        (input_tokens, output_tokens, txt_mask) = self._maybe_sub_sample(batch["input_ids"], batch['mask'])
        
        attention_mask = torch.cat([batch['mask'],batch['img_mask']], dim=-1)
        continuous_mode = self.config.trainer.image_mode == "continuous"
        joint_ar_nar_mask, modality = None, None
        unet_conditioning, xt, x0, x_img_emb, modality_mask = None, None, input_tokens, None, None

        if self.parameterization != "ar":
            t = self._sample_t(x0.shape[0], x0.device, epoch)
            if self.T > 0:
                t = (t * self.T).to(torch.int)
                t = t / self.T
                t += 1 / self.T # t \in {1/T, 2/T, ..., 1}

            if self.change_of_variables:
                unet_conditioning = t[:, None]
                f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
                f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
                move_chance = torch.exp(f_0 + t * (f_T - f_0))
                move_chance = move_chance[:, None]
            else:
                # total, rate
                sigma, dsigma = self.noise(t)
                unet_conditioning = sigma[:, None]
                move_chance = 1 - torch.exp(-sigma[:, None])

            xt, ignore_batch_mask_for_metrics, joint_ar_nar_mask, should_mask_txt, should_mask_img, move_indices = \
                self.q_xt(x0, move_chance, return_ignore_batch_mask_for_metrics=True, batch=batch)
    
        true_logits = None
        #print(f'xt:{xt.shape}')
        model_output = self.forward(xt, unet_conditioning, batch=batch)

        if isinstance(model_output, tuple):
            model_output, true_logits = model_output

        to_dtype = self.dtype if self.config.trainer.low_precision_loss else torch.float32
        model_output = model_output.to(to_dtype)
        if true_logits is not None:
            true_logits = true_logits.to(self.dtype)

        if not self.is_compiled:
            utils.print_nans(model_output, "model_output")

        diffusion_loss = None
        if self.T > 0:
            diffusion_loss = self._d3pm_loss(model_output=model_output, xt=xt, x0=x0, t=t)

        log_p_theta = torch.gather(input=model_output, dim=-1, index=x0[:, :, None]).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        std_weighting = (dsigma / torch.expm1(sigma))[:, None]

        loss = -log_p_theta * std_weighting

        if diffusion_loss is not None:
            assert self.T > 0
            loss = diffusion_loss

        std_loss = -log_p_theta * std_weighting
        loss_dict = dict(std_loss=std_loss.detach(), extra_losses=dict())

        if self.config.trainer.log_seperate_modal_losses:
            assert not continuous_mode
            loss_dict.update(
                dict(
                    #std_txt_loss=(std_loss.detach() * modality_mask[..., 0] * attention_mask), 
                    #std_img_loss=(std_loss.detach() * modality_mask[..., 1] * attention_mask)
                    std_txt_loss = (std_loss.detach()),
                    std_img_loss = (std_loss.detach())
                )
            )

        #if getattr(self.config.trainer, "mask_entire_modality", None) is not None and self.backbone.training and not self.config.parameterization == "ar":
        #    loss_dict['batch_ignore_loss'] = ignore_batch_mask_for_metrics.squeeze(-1)

        #if joint_ar_nar_mask is not None:
        #    if "batch_ignore_loss" in loss_dict:
        #        loss_dict["batch_ignore_loss"] = loss_dict["batch_ignore_loss"] | joint_ar_nar_mask
        #    else:
        #        loss_dict["batch_ignore_loss"] = joint_ar_nar_mask

        if joint_ar_nar_mask is not None:
            pass # Defer loss mean until after ar_loss is calculated
        else:
            #log_probs = model_output  # 即 _subs_parameterization 的输出
            #probs = log_probs.exp()
            #entropy = - (probs * log_probs).sum(dim=-1)  # [B, T]
            #mask = (xt == self.mask_index)
            #div_loss = -entropy[mask].mean()
            #print(f'div_loss:{div_loss.detach().cpu().item()}')
            _attention_mask = torch.ones_like(attention_mask) if getattr(self.config.trainer, "force_full_attention_mask_loss_only", False) else attention_mask
            loss = (loss * _attention_mask).sum() / _attention_mask.sum()
            #loss = loss +  1e-2 * div_loss
            
            #loss = torch.nan_to_num(loss, nan=0.0)
            #if torch.isnan(loss).any():
            #     print(f"Warning: NaN loss at step {step}, skipping this batch")
            #     continue

        loss_dict = dict(loss=loss, **loss_dict)
        std_loss = loss_dict.get("std_loss", 0)
        #std_nlls = std_loss * attention_mask
        std_nlls = std_loss
        if "batch_ignore_loss" in loss_dict:
            attention_mask = torch.where(loss_dict['batch_ignore_loss'][:, None].repeat(1, attention_mask.shape[-1]), torch.full_like(attention_mask, False), attention_mask)
            
        losses = Loss(
            loss=loss_dict["loss"],
            img_loss=loss_dict.get("img_loss", 0),
            txt_loss=loss_dict.get("txt_loss", 0),
            nlls=std_nlls,
            txt_nlls=loss_dict.get("std_txt_loss", 0),
            img_nlls=loss_dict.get("std_img_loss", 0),
            token_mask=attention_mask,
            modality_mask=modality_mask,
            extra_losses=loss_dict.get("extra_losses", None),
        )

        if prefix == "train":
            return losses
        elif prefix == "val":
            #print(losses.token_mask.device)
            #self.valid_metrics.update(losses.nlls, losses.token_mask)
            return losses

        
    @torch.no_grad()
    def zero_shot_eval(self):
        dataloader = self.validation_dataloader
        total_batches = len(dataloader)
        rprint(f"Zero shot eval with {total_batches} batches with limit_val_batches: {self.config.trainer.limit_val_batches}")
        for idx, batch in tqdm(enumerate(dataloader), total=total_batches, desc="Zero shot eval validation steps", disable=not is_main_process()):
            if self.config.trainer.limit_val_batches is not None and idx >= self.config.trainer.limit_val_batches:
                break
            self.zero_shot_eval_step(batch, idx)
        
        self.zero_shot_eval_epoch_end()

    def validate(self, name=None):
        self.on_validation_epoch_start()

        total_batches = 1
        _dataloader = self.validation_dataloader
        rprint(f"Validating with {total_batches} batches on {self.world_size} GPUs with batch size {self.config.loader.eval_batch_size}")
        for idx, batch in tqdm(enumerate(_dataloader), total=total_batches, desc="Validation steps", disable=not is_main_process()):
            #if idx >= total_batches:
            #    break
            self.validation_step(batch, idx, idx, name)
        # batch = next(iter(_dataloader))
        #
        # if self.config.mode == "eval":
        #     gprint(f"Batch shape: {batch['input_ids'].shape}")
        #
        # self.on_validation_epoch_end(example_batch=batch)
        self.on_validation_epoch_cleanup()

    @cached_property
    def global_batch_size(self):
        """Batch size for a single step over all GPUs"""
        # SPMD treats all ranks [regardless of node] as a single device
        return self.step_batch_size * self.world_size

    @cached_property
    def step_batch_size(self):
        """Batch size for a single step for a single GPU"""
        return self.config.loader.batch_size * self.config.trainer.accumulate_grad_batches

    @cached_property
    def world_size(self):
        """Number of GPUs over all nodes"""
        return get_world_size()

    @cached_property
    def num_tokens_per_sample(self):
        """Number of tokens per sample"""
        return self.config.model.length

    @cached_property
    def gradient_accumulation_steps(self):
        """Number of gradient accumulation steps"""
        return self.config.trainer.accumulate_grad_batches

    @cached_property
    def static_txt_sl(self):
        return slice(None, self.config.model.txt_length)

    @cached_property
    def static_img_sl(self):
        return slice(-self.config.model.img_length, None)

    def img_txt_pair_batch_mask(self, batch=None):
        return batch["modality_mask"][..., 1].sum(dim=-1) > 0

    def txt_sl(self, batch=None):
        return batch["modality_mask"][..., 0]

    def img_sl(self, batch=None):
        return batch["modality_mask"][..., 1]

    @cached_property
    def is_compiled(self):
        return False
    
    @property
    def allow_slicing(self):
        return False

    @property
    def training(self):
        return self.backbone.training

    def get_step_metrics(self):
        return {
            "trainer/global_step": self.global_step,
            "global_samples": self.global_step * self.global_batch_size,
            "train_metrics/global_tokens": self.global_step * self.global_batch_size * self.config.model.length,
            "effective_global_tokens": self.global_step * self.global_batch_size * self.config.model.length * (0.5 if self.config.parameterization == "subs" else 1.0),
            "effective_global_step": int(self.global_step * (0.5 if self.config.parameterization == "subs" else 1.0)),
        }

    def train(self):
        tr = self.config.trainer
        total_batch_size = self.global_batch_size
        initial_global_step = self.global_step
        true_step = 0
        first_epoch = 0
        self.current_run_global_step = 0
        self.current_run_fwd_bwd_pass = 0
        rprint(f"Started at step {self.accelerator.step}")

        # There is an unknown bug with accelerator where non-master ranks don't load the step count from a checkpoint.
        # We workaround by broadcasting the step count if necessary
        if is_torch_cuda_available():
            dprint(f"Gathering step from {self.world_size} ranks")
            starting_steps = gather_object([self.accelerator.step])
            rprint(f"Starting steps: {starting_steps}")
            if not all([x > 0 for x in starting_steps]):
                rprint(f"Not all ranks have >0 step, setting to: {starting_steps[0]}")
                self.accelerator.step = starting_steps[0]

        rprint(f"***** Starting training at global step: {0} *****")
        rprint(f"  Instantaneous batch size per device = {self.config.loader.batch_size}")
        rprint(f"  Gradient Accumulation steps = {tr.accumulate_grad_batches}")
        rprint(f"  Num GPUs = {tr.devices}")
        rprint(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        rprint(f"  Total optimization steps = {tr.max_steps}")
        rprint(f"  Reported Global Batch Size: {self.global_batch_size}, Reported Step Batch Size: {self.step_batch_size}, Reported World Size: {self.world_size}")

        num_epoch_steps = len(self.train_dataloader)
        rprint(f"  Num examples = {len(self.train_dataloader.dataset)}")
        rprint(f"  Num batches each epoch = {len(self.train_dataloader)}")
        rprint(f"Train Dataloader Size on single GPU: {num_epoch_steps}")
        # if len(self.train_dataloader.dataset) < total_batch_size:
        #     rprint("The training dataloader is smaller than the total batch size. This may lead to unexpected behaviour.")


        progress_bar = tqdm(range(0, tr.max_steps), initial=initial_global_step, desc="Steps", disable=not is_local_main_process(), leave=False, smoothing=0.15)

        global_step_metrics = defaultdict(float)
        accumulate_steps = 0
        first_start_time = time.time()
        # self.on_train_start()

        rprint(f"Training for {tr.num_epochs} epochs...")
        last_end_step_time = start_timing(f"Dataloading accum:{accumulate_steps}, #{true_step}, global_step:{self.global_step}")
        for epoch in range(first_epoch, tr.num_epochs):
            rprint(f"Starting epoch {epoch}...")
            for step, batch in enumerate(self.train_dataloader):
                ddprint(f"Data Step: {step}")
                global_step_metrics[f"dataloading_time"] += end_timing(last_end_step_time)
                if batch is None:
                    rprint(f"Batch is None at step {step}")
                    continue
                with self.accelerator.accumulate(self.backbone):
                    ddprint(f"Before forward pass for global_step: {self.global_step}")
                    start_forward_time = start_timing(f"Forward Pass accum:{accumulate_steps}, #{true_step}, global_step:{self.global_step}")
                    global_step_metrics["examples_seen_per_gpu"] += len(next(iter(batch.values())))
                    state: TrainingState = TrainingState(
                        epoch_step=step,
                        num_epoch_steps=num_epoch_steps,
                        global_step=self.global_step,
                        epoch=epoch,
                        true_step=true_step,
                        current_run_global_step=self.current_run_global_step,
                    )
                    ddprint(f"Before Fwd: {step}")
                    with nullcontext():
                        losses = self.training_step(batch, step, epoch)

                    ddprint(f"After Fwd: {step}")
                    #global_step_metrics["forward_pass_time"] += end_timing(start_forward_time)
                    true_step += 1

                    if isinstance(losses, Loss):
                        #print('losses is loss')
                        loss = losses.loss
                        
                        if torch.isnan(loss).any():
                            print("Loss contains NaN, skipping this batch.")
                            continue

                        global_step_metrics["loss"] = loss.detach().cpu().item()
                        # metrics = self.train_metrics(losses.nlls, losses.token_mask)
                        def evaluate_extra_log_data():
                            return {}
                        ddprint(f"Before loss: {step}")

                    ddprint(f"Before backward pass for global_step: {self.global_step}")

                    # Short-circuit to avoid XLA eval
                    if torch.isfinite(loss).all():
                        start_backward_time = start_timing(f"Backward Pass accum:{accumulate_steps}, #{true_step}, global_step:{self.global_step}")
                        if self.accelerator.sync_gradients:
                            start_sync_time = start_timing(f"Gradient Sync global_step:{self.global_step}")

                        # After each fwd, we perform a bwd. However, if we are accumulating there is an internal no_sync so the gradients remain on the GPU until
                        # the final bwd before a step. This can be controlled by sync_each_batch. Note that for the last bwd, the sync happens inside the bwd call below, so any timing for stragglers needs to happen before this call.
                        with nullcontext():
                            ddprint(f"Before accelerator.backward for global_step: {self.global_step}")
                            self.accelerator.backward(loss)
                            ddprint(f"After accelerator.backward for global_step: {self.global_step}")

                        with nullcontext():
                            if self.accelerator.sync_gradients:
                                ddprint(f"Before after.backward for global_step: {self.global_step}")
                                self.after_backward(None)
                                if tr.gradient_clip_val is not None:
                                    ddprint(f"Before self.accelerator.clip_grad_norm_ for global_step: {self.global_step}")
                                    total_grad_norm = self.accelerator.clip_grad_norm_(self.backbone.parameters(), tr.gradient_clip_val)
                                    ddprint(f"After self.accelerator.clip_grad_norm_ for global_step: {self.global_step}")

                        with nullcontext():
                            ddprint(f"Before optimizer step for global_step: {self.global_step}, {step}")
                            self.optimizer.step()
                            ddprint(f"After optimizer step for global_step: {self.global_step}, {step}")
                            self.lr_scheduler.step()
                            ddprint(f"After lr_scheduler step for global_step: {self.global_step}, {step}")

                        zero_grad_kwargs = dict()
                        if "apex" not in self.config.trainer.optimizer_cls:
                            zero_grad_kwargs["set_to_none"] = tr.set_grads_to_none

                        ddprint(f"Before zero_grad for global_step: {self.global_step}, {step}")
                        self.optimizer.zero_grad(**zero_grad_kwargs)
                        ddprint(f"Zeroed gradients for global_step: {self.global_step}, {step}")

                        if self.accelerator.sync_gradients:
                            if self.ema is not None:
                                self.ema.step(self.get_params())
                            #global_step_metrics["gradient_sync_time"] += end_timing(start_sync_time)

                        #global_step_metrics["backward_pass_time"] += end_timing(start_backward_time)
                    else:
                        if not torch.isfinite(loss).all(): print(f"Loss is not finite: {loss}")
                        print("Skipping backward pass!")

                    accumulate_steps += 1
                    self.current_run_fwd_bwd_pass += 1

                # Important: A single "global_step" is a single optimizer step. The accumulate decorator silently skips backward + optimizer to allow for gradient accumulation.
                # A "true_step" counts the number of forward passes (on a per-GPU basis). The condition below should only happen immediately after a backward + optimizer step.
                ddprint(f"Syncing gradients for global_step: {self.global_step}. Should sync: {self.accelerator.sync_gradients}, {step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")
                if self.accelerator.sync_gradients:
                    start_gradient_sync_time = start_timing(f"On Sync Gradients global_step:{self.global_step}, {step}")

                    #print(f"Before on_train_step_end for global_step: {self.global_step}, {step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")
                    del loss, losses, batch
                    gradient_sync_time_after_train_step_end_time = start_timing(f"On Sync Gradients global_step:{self.global_step}, {step}")
                    self.on_train_step_end(self.lr_scheduler,state)
                    ddprint(f"After on_train_step_end for global_step: {self.global_step}, {step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")
                    #global_step_metrics["gradient_sync_time_after_train_step_end"] += end_timing(gradient_sync_time_after_train_step_end_time)

                    progress_bar.update(1)
                    self.global_step += 1
                    self.current_run_global_step += 1
                    #global_step_metrics["gradient_sync_time"] += end_timing(start_gradient_sync_time)

                    logs = {
                        #"examples_seen": self.global_step * total_batch_size,
                        "trainer/global_step": self.global_step,
                        **{k: v for k, v in global_step_metrics.items()},
                        **{f"lr_{i}": lr for i, lr in enumerate(self.lr_scheduler.get_last_lr())}
                    }

                    progress_bar.set_postfix(**logs)

                    global_step_metrics = defaultdict(float)
                    accumulate_steps = 0

                    # if self.global_step >= tr.max_steps:
                    #     break

                    ddprint(f"After logging for step v3: {self.global_step}, {step}")

                    ddprint(f"After logging for step v4: {self.global_step}, {step}")
                    ddprint(f"Finished sync_gradients: {self.global_step}, {self.accelerator.step}, {self.accelerator.gradient_accumulation_steps}")

                ddprint(f"Finished step: {self.global_step},{step},{self.accelerator.step},{self.accelerator.gradient_accumulation_steps},{self.accelerator.gradient_state.__repr__()}")
                last_end_step_time = start_timing(f"Dataloading #{true_step + 1}")

            # if self.global_step >= tr.max_steps:
            #     break

            dprint(f"Finished epoch: {epoch}")

        # Create the pipeline using using the trained modules and save it.
        rprint("Training finished.")
        barrier()

        if tr.profile_memory:
            print_memory(verbose=True)
            save_memory_profile(self.config.output_dir / "profile")

        if self.global_step > 100 or tr.skip_early_checkpointing is False:
            self.checkpoint(state)

        barrier()
