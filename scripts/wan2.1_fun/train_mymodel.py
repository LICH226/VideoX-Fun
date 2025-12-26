#!/usr/bin/env python
# coding=utf-8

import argparse
import gc
import logging
import math
import os
import shutil
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
import diffusers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor

from videox_fun.data.tryon_video import TryOnDataset 
from videox_fun.data.bucket_sampler import AspectRatioBatchImageVideoSamplerTryOn, ASPECT_RATIO_512
from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.models.wan_transformer3d_tryon import WanTransformer3DTryonModel
from videox_fun.pipeline import WanFunInpaintPipeline 
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import save_videos_grid
from videox_fun.utils.lora_utils import create_network

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

# -------------------------------------------------------------------------
# 2. Utils
# -------------------------------------------------------------------------

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    return initial_value + step_size * current_step

# -------------------------------------------------------------------------
# 3. Validation Logic (Simplified for VTON)
# -------------------------------------------------------------------------
def log_validation(vae, text_encoder, tokenizer, clip_image_encoder, transformer3d, args, config, accelerator, weight_dtype, global_step):
    try:
        logger.info("Running validation... ")

        # Load validation model copy
        transformer3d_val = WanTransformer3DTryonModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)
        transformer3d_val.load_state_dict(accelerator.unwrap_model(transformer3d).state_dict())
        
        scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        # 强制使用 Inpaint Pipeline
        pipeline = WanFunInpaintPipeline(
            vae=accelerator.unwrap_model(vae).to(weight_dtype), 
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            transformer=transformer3d_val,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        pipeline = pipeline.to(accelerator.device)

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

        # 这里简化验证逻辑：VTON 通常需要成对的 (Model Image + Cloth Image)
        # 如果 args.validation_prompts 只是文本，效果可能不好。
        # 理想情况下，这里应该加载几个固定的 Validation Case (Input, Cloth, Mask)
        # 暂时保留原逻辑，但在 VTON 中可能需要后续修改为读取特定文件夹的验证集
        
        # ... (Validation Loop Logic Placeholder) ... 
        # 由于 VTON 需要特定的 input/mask 输入，单纯 text2video 的 validation 无法工作。
        # 建议后续专门写一个 validation 脚本，或者在这里硬编码几个测试样本路径。
        
        del pipeline
        del transformer3d_val
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Eval error: {e}")



def main():
    # --- A. Parse Args (Pruned) ---
    parser = argparse.ArgumentParser(description="VTON Training Script")
    
    # Model & Config
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--image_encoder_path", type=str, default=None)
    parser.add_argument("--image_encoder_2_path", type=str, default=None)
    
    # Data
    parser.add_argument("--train_data_dir", type=str, required=True, help="Root of VIVID/VITON/DressCode")
    parser.add_argument("--train_data_meta", type=str, required=True, help="JSONL file path")
    parser.add_argument("--video_sample_size", type=int, default=512, help="Width (Height is usually 384 or adapted)")
    parser.add_argument("--video_sample_n_frames", type=int, default=49)
    parser.add_argument("--video_repeat", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading")
    parser.add_argument("--video_sample_stride", type=int, default=1, help="Stride for video sampling")

    # Training Hyperparams
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="The scheduler type to use")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Advanced
    parser.add_argument("--use_deepspeed", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trainable_modules", nargs='+', default=["attn", "norm"]) # Adjust based on strategy
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")

    # Loss Params
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--motion_sub_loss", action="store_true", help="Whether to use motion sub loss")
    parser.add_argument("--motion_sub_loss_ratio", type=float, default=0.0, help="Ratio for motion sub loss")
    
    # Lora 
    parser.add_argument("--rank", type=int, default=128, help="The dimension of the LoRA update matrices")
    parser.add_argument("--network_alpha", type=float, default=64, help="The alpha parameter for LoRA scaling")
    parser.add_argument("--lora_skip_name", type=str, default=None, help="List of module names to skip for LoRA")
    
    #evaluation
    parser.add_argument("--validation_paths", type=str, default=None, help="Path to validation dataset metadata")
    parser.add_argument("--validation_steps", type=int, default=500, help="Run validation every X steps")

    args = parser.parse_args()
    
    # --- B. Accelerator Setup ---
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Logging setup
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed)

    # --- C. Load Models ---
    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": weight_dtype = torch.bfloat16

    # 1. Tokenizer & Text Encoder
    logger.info("load text_encoder!")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=weight_dtype
    ).eval()

    # 2. VAE
    logger.info("load VAE!")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).eval()
    
    # 3. CLIP Image Encoder (For Cloth condition)
    logger.info("load clip_encoder!")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).eval()

    # 4. Transformer (Trainable)
    logger.info("load dit!")
    transformer3d = WanTransformer3DTryonModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True    
    )

    logger.info(f"=> loading siglip_image_encoder: {args.image_encoder_path}")
    siglip_image_encoder = SiglipVisionModel.from_pretrained(args.image_encoder_path)
    siglip_image_processor = SiglipImageProcessor.from_pretrained(args.image_encoder_path)
    siglip_image_encoder.eval()
    
    logger.info(f"=> loading dino_image_encoder: {args.image_encoder_2_path}")
    dino_image_encoder = AutoModel.from_pretrained(args.image_encoder_2_path)
    dino_image_processor = AutoImageProcessor.from_pretrained(args.image_encoder_2_path)
    dino_image_encoder.eval()
    dino_image_processor.crop_size = dict(height=384, width=384)
    dino_image_processor.size = dict(shortest_edge=384)

    # Freeze Frozen Models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    siglip_image_encoder.requires_grad_(False)
    dino_image_encoder.requires_grad_(False)

    transformer3d.train()
    logger.info("🧹 Zero-initializing IP-Adapter modules (k_ip, v_ip)...")
    for name, module in transformer3d.named_modules():
        if isinstance(module, torch.nn.Linear) and ("k_ip" in name or "v_ip" in name):
            with torch.no_grad():
                if hasattr(module, "weight"):
                    torch.nn.init.zeros_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    logger.info("✅ Zero initialization complete.")
    transformer3d.requires_grad_(False)


    trainable_keywords = [
        "patch_embedding", 
        "subject_image_proj_model", 
        "k_ip", 
        "v_ip", 
        "norm_k_ip"
    ]

    for name, param in transformer3d.named_parameters():
        if any(keyword in name for keyword in trainable_keywords):
            param.requires_grad = True

    transformer3d.to(weight_dtype)

    # EMA Setup
    ema_transformer3d = None
    if args.use_ema:
        ema_transformer3d = EMAModel(transformer3d.parameters(), model_cls=WanTransformer3DModel, model_config=transformer3d.config)
        ema_transformer3d.to(accelerator.device)

    # --- D. Optimizer & Scheduler ---
    optimizer_cls = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_count = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in transformer3d.parameters())
    logger.info(f"🔥🔥🔥 Final Trainable Parameters: {trainable_params_count / 1e6:.2f} M / {total_params / 1e6:.2f} M")
    transformer3d.enable_gradient_checkpointing() if args.gradient_checkpointing else None
    optimizer = optimizer_cls(trainable_params, lr=args.learning_rate)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs'])))

    # --- E. Dataset & Dataloader (Crucial Change for VTON) ---
    print(f"Initializing VTON Dataset...")
    # 这里直接使用我们自定义的 Dataset，移除了 Bucket 逻辑
    train_dataset = TryOnDataset(
        ann_path=args.train_data_meta,
        data_root=args.train_data_dir,
        video_sample_size=args.video_sample_size, 
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        image_sample_size=args.video_sample_size,
        video_repeat=args.video_repeat,
        text_drop_ratio=0.0
    )

    def vton_collate_fn(examples):
        # 1. 随机分辨率 (256-512, 步长32)
        target_res = random.choice(range(256, 512 + 1, 32))
        
        # 2. 判断 Batch 类型 (全视频 or 全图片)
        # 取第一个样本看长度，>1 就是视频
        first_len = examples[0]["pixel_values"].shape[0]
        is_video = first_len > 1
        
        # 3. 设定目标长度
        if is_video:
            # 如果是视频，统一对齐到当前batch最大的帧数 (通常49)，并满足 4k+1
            max_len = max([ex["pixel_values"].shape[0] for ex in examples])
            target_len = (max_len - 1) // 4 * 4 + 1
            target_len = max(1, int(target_len))
        else:
            target_len = 1 # 图片固定为1

        temporal_keys = ["pixel_values", "densepose_pixel_values", "agnostic_pixel_values", "mask_pixel_values"]
        aligned_examples = []
        
        for ex in examples:
            new_ex = {}
            curr_len = ex["pixel_values"].shape[0]
            new_ex["data_type"] = "video" if is_video else "image"

            for key in temporal_keys:
                tensor = ex[key] 
                
                if curr_len < target_len:
                    repeat_times = target_len // curr_len + 1
                    tensor = tensor.repeat(repeat_times, 1, 1, 1)[:target_len]
                elif curr_len > target_len:
                    tensor = tensor[:target_len]
                
                mode = 'nearest' if 'mask' in key or 'densepose' in key else 'bilinear'
                align_corners = False if mode != 'nearest' else None 

                tensor_resized = F.interpolate(
                    tensor, 
                    size=(target_res, target_res), 
                    mode=mode, 
                    align_corners=align_corners,
                    antialias=True if mode != 'nearest' else False
                )
                new_ex[key] = tensor_resized

            new_ex["cloth_pixel_values"] = F.interpolate(
                ex["cloth_pixel_values"], size=(target_res, target_res), 
                mode='bilinear', align_corners=False, antialias=True
            )
            new_ex["text"] = ex["text"]
            aligned_examples.append(new_ex)

        # Stack
        batch = {
            "pixel_values": torch.stack([ex["pixel_values"] for ex in aligned_examples]),
            "cloth_pixel_values": torch.stack([ex["cloth_pixel_values"] for ex in aligned_examples]),
            "agnostic_pixel_values": torch.stack([ex["agnostic_pixel_values"] for ex in aligned_examples]),
            "mask_pixel_values": torch.stack([ex["mask_pixel_values"] for ex in aligned_examples]),
            "densepose_pixel_values": torch.stack([ex["densepose_pixel_values"] for ex in aligned_examples]),
            "text": [ex["text"] for ex in aligned_examples],
            "data_type": [ex["data_type"] for ex in aligned_examples]
        }
        
        # Tokenizer
        prompt_ids = tokenizer(
            batch["text"], max_length=512, padding="max_length", truncation=True, return_tensors="pt"
        )
        batch['input_ids'] = prompt_ids.input_ids
        batch['attention_mask'] = prompt_ids.attention_mask
        
        return batch

    base_sampler = RandomSampler(train_dataset)
    
    # 2. 分组采样 (使用新 JSONL 的宽高)
    batch_sampler = AspectRatioBatchImageVideoSamplerTryOn(
        sampler=base_sampler,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        aspect_ratios=ASPECT_RATIO_512,
        drop_last=False
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=vton_collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    transformer3d, optimizer, train_dataloader = accelerator.prepare(
        transformer3d, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        # --- 修复开始：清洗 config 字典 ---
        tracker_config = {}
        for k, v in vars(args).items():
            # TensorBoard 只接受 int, float, str, bool
            if isinstance(v, (int, float, str, bool)):
                tracker_config[k] = v
            elif isinstance(v, list):
                # 将列表转换为字符串，例如 ['attn', 'norm'] -> "['attn', 'norm']"
                tracker_config[k] = str(v)
            elif v is None:
                tracker_config[k] = "None"
            else:
                # 其他复杂对象也强转为字符串
                tracker_config[k] = str(v)
        # -------------------------------
        
        accelerator.init_trackers("wan_vton_project", config=tracker_config)

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)
    dino_image_encoder.to(accelerator.device, dtype=weight_dtype)
    siglip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # --- G. Training Loop ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    initial_global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            if os.path.exists(os.path.join(args.output_dir, path)):
                checkpoint_path = os.path.join(args.output_dir, path)
            else:
                checkpoint_path = args.resume_from_checkpoint
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            if path is None:
                checkpoint_path = None
            else:
                checkpoint_path = os.path.join(args.output_dir, path)

        if checkpoint_path is not None:
            accelerator.print(f"Resuming from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            
            try:
                global_step = int(os.path.basename(checkpoint_path).split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
            except ValueError:
                accelerator.print("Could not calculate global step, starting from 0")

    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps
    )
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(initial_global_step)

    def encode_vae(video_tensor):
        return vae.encode(video_tensor)[0].sample()

    log_buffer = {"video": 0, "image": 0, "total": 0}
    for epoch in range(first_epoch, args.num_train_epochs):
        train_dataloader_iterator = train_dataloader
        if args.resume_from_checkpoint and epoch == first_epoch and initial_global_step > 0:
            resume_step = (initial_global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps
            
            if resume_step > 0:
                train_dataloader_iterator = accelerator.skip_first_batches(train_dataloader, resume_step)
                accelerator.print(f"Skipping {resume_step} batches in epoch {epoch}...")
        
        for step, batch in enumerate(train_dataloader_iterator):
            current_type = batch['data_type'][0] 
            log_buffer[current_type] += 1
            log_buffer["total"] += 1
            with accelerator.accumulate(transformer3d):
                pixel_values = batch["pixel_values"].to(weight_dtype).transpose(1, 2)
                densepose_values = batch["densepose_pixel_values"].to(weight_dtype).transpose(1, 2)
                cloth_values = batch["cloth_pixel_values"].to(weight_dtype).transpose(1, 2)
                agnostic_values = batch["agnostic_pixel_values"].to(weight_dtype).transpose(1, 2)
                mask_values = batch["mask_pixel_values"].to(weight_dtype).transpose(1, 2)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch['input_ids'].to(accelerator.device), attention_mask=batch['attention_mask'].to(accelerator.device))[0]

                with torch.no_grad():
                    latents = encode_vae(pixel_values) 
                    mask_latents = encode_vae(agnostic_values)
                    densepose_latents = encode_vae(densepose_values) 

                    def prepare_vton_mask(mask, latents):
                        mask_padded = torch.cat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        b, c, t, h, w = mask_padded.shape
                        mask_view = mask_padded.view(b, c, t // 4, 4, h, w)
                        mask_folded = mask_view.permute(0, 3, 1, 2, 4, 5).reshape(b, 4, t // 4, h, w)
                        target_h, target_w = latents.shape[-2], latents.shape[-1]
                        mask_final = F.interpolate(
                            mask_folded, 
                            size=(mask_folded.shape[2], target_h, target_w), 
                            mode="nearest"
                        )
                        
                        return mask_final                    
                    
                    def encode_siglip_image_emb(siglip_image, device, dtype):
                        siglip_image = siglip_image.to(device, dtype=dtype)
                        res = siglip_image_encoder(siglip_image, output_hidden_states=True)

                        siglip_image_embeds = res.last_hidden_state

                        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
                        
                        return siglip_image_embeds, siglip_image_shallow_embeds

                    def encode_dinov2_image_emb(dinov2_image, device, dtype):
                        dinov2_image = dinov2_image.to(device, dtype=dtype)
                        res = dino_image_encoder(dinov2_image, output_hidden_states=True)

                        dinov2_image_embeds = res.last_hidden_state[:, 1:]

                        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)

                        return dinov2_image_embeds, dinov2_image_shallow_embeds

                    def encode_image_emb(image_tensor, device, dtype):
                        """
                        Args:
                            image_tensor: (B, C, H, W) 范围 [-1, 1] (来自 Dataset 的原始 tensor)
                        """
                        image_01 = image_tensor * 0.5 + 0.5
                        
                        # 辅助函数：处理 Processor 归一化 (PyTorch Tensor 方式)
                        def process_with_processor(images, processor):
                            # images: (B, C, H, W) 范围 [0, 1]
                            # processor 仅负责 ImageNet Normalize (减均值除方差)
                            if isinstance(images, torch.Tensor) and images.dtype == torch.bfloat16: images = images.to(torch.float32)
                            return processor(
                                images=images,
                                return_tensors="pt",
                                do_resize=False,       # 我们自己做 interpolate
                                do_center_crop=False,  # 我们自己做 crop
                                do_rescale=False,      # 输入已经是 [0, 1] float，不需要 /255
                                do_normalize=True      # 执行标准化
                            ).pixel_values.to(device, dtype)

                        # Step B: 准备 Low Res (384x384)
                        # 对应 PIL.resize((384, 384))
                        image_low_res = F.interpolate(image_01, size=(384, 384), mode='bicubic', align_corners=False, antialias=True)
                        
                        # Step C: 准备 High Res (768x768) 并切分
                        # 对应 PIL.resize((768, 768))
                        image_high_res_full = F.interpolate(image_01, size=(768, 768), mode='bicubic', align_corners=False, antialias=True)
                        
                        # 切分成 4 个 384x384 的块 (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
                        # Tensor slicing: [..., h_start:h_end, w_start:w_end]
                        crops = [
                            image_high_res_full[:, :, 0:384, 0:384],
                            image_high_res_full[:, :, 0:384, 384:768],
                            image_high_res_full[:, :, 384:768, 0:384],
                            image_high_res_full[:, :, 384:768, 384:768],
                        ]
                        # 堆叠成 batch: (B, 4, C, 384, 384) -> Flatten -> (B*4, C, 384, 384)
                        image_high_res_crops = torch.stack(crops, dim=1)
                        nb_split_image = 4
                        image_high_res_flat = rearrange(image_high_res_crops, 'b n c h w -> (b n) c h w')

                        # Step D: 计算 Low Res 特征
                        siglip_low_input = process_with_processor(image_low_res, siglip_image_processor)
                        dino_low_input = process_with_processor(image_low_res, dino_image_processor)
                        
                        siglip_embeds_low, siglip_shallow_low = encode_siglip_image_emb(siglip_low_input, device, dtype)
                        dinov2_embeds_low, dinov2_shallow_low = encode_dinov2_image_emb(dino_low_input, device, dtype)
                        
                        image_embeds_low_res_deep = torch.cat([siglip_embeds_low, dinov2_embeds_low], dim=2)
                        image_embeds_low_res_shallow = torch.cat([siglip_shallow_low, dinov2_shallow_low], dim=2)

                        # Step E: 计算 High Res 特征
                        siglip_high_input = process_with_processor(image_high_res_flat, siglip_image_processor)
                        dino_high_input = process_with_processor(image_high_res_flat, dino_image_processor)
                        
                        siglip_embeds_high, _ = encode_siglip_image_emb(siglip_high_input, device, dtype)
                        dinov2_embeds_high, _ = encode_dinov2_image_emb(dino_high_input, device, dtype) # DINOv2 可能返回切片后的特征
                        
                        # Reshape back: (B*4, L, D) -> (B, 4*L, D)
                        siglip_high_res_deep = rearrange(siglip_embeds_high, '(b n) l c -> b (n l) c', n=nb_split_image)
                        dinov2_high_res_deep = rearrange(dinov2_embeds_high, '(b n) l c -> b (n l) c', n=nb_split_image)
                        
                        image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)

                        return dict(
                            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
                            image_embeds_low_res_deep=image_embeds_low_res_deep,
                            image_embeds_high_res_deep=image_embeds_high_res_deep,
                        )

                    mask_values = prepare_vton_mask(mask_values, latents)
                    inpaint_latents = torch.cat([mask_values, mask_latents, densepose_latents], dim=1) 
                    
                    clip_context = clip_image_encoder(cloth_values)

                    cloth_img_frame0 = cloth_values[:, :, 0, :, :] 
                    subject_image_embeds_dict = encode_image_emb(
                        cloth_img_frame0, 
                        accelerator.device, 
                        weight_dtype
                    )

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,)).long()
                timesteps = noise_scheduler.timesteps[indices].to(latents.device)
                sigmas = noise_scheduler.sigmas[indices].flatten().to(latents.device)
                while len(sigmas.shape) < latents.ndim: sigmas = sigmas.unsqueeze(-1)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents 
                
                target_shape = noisy_latents.shape
                seq_len = math.ceil((target_shape[3] * target_shape[4]) / (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) * target_shape[2])
                
                noisy_latents = noisy_latents.to(weight_dtype)
                encoder_hidden_states = encoder_hidden_states.to(weight_dtype)
                clip_context = clip_context.to(weight_dtype)
                inpaint_latents = inpaint_latents.to(weight_dtype)


                with accelerator.autocast():
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        seq_len=seq_len,
                        context=encoder_hidden_states,
                        t=timesteps,
                        y=inpaint_latents,   
                        clip_fea=clip_context,
                        subject_image_embeds_dict=subject_image_embeds_dict,
                    )

                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                # 6. Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                current_batch_frames = pixel_values.shape[2] 
                current_batch_res = pixel_values.shape[3]    
                log_data = {
                    "train/loss": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "debug/type": 1.0 if current_type == 'video' else 0.0, # 1=Video, 0=Image
                    "debug/frames": current_batch_frames,
                    "debug/resolution": current_batch_res,
                    "debug/video_ratio": log_buffer['video'] / log_buffer['total'] # 累计视频比例
                }
                
                accelerator.log(log_data, step=global_step)
                
                if step % 10 == 0: # 每10步打印一次，防止刷屏
                    logger.info(f"Step {global_step}: Type={current_type.upper()}, Res={current_batch_res}, Frames={current_batch_frames}, Loss={loss.item():.4f}")

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()