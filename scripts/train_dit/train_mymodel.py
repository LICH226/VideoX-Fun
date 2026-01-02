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
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor, AutoTokenizer

from videox_fun.data.tryon_video import TryOnDataset 
from videox_fun.data.bucket_sampler import AspectRatioBatchImageVideoSamplerTryOn, ASPECT_RATIO_512
from videox_fun.models import AutoencoderKLWan3_8, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
from videox_fun.models.wan_transformer3d_tryon import WanTransformer3DTryonModel
from peft import LoraConfig, get_peft_model, PeftModel
from functools import partial
from videox_fun.pipeline.pipeline_tryon_wan2_2 import WanTryOnPipeline
from PIL import Image
import wandb

logger = get_logger(__name__, log_level="INFO")

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
    parser.add_argument("--data_dir", type=str, required=True, help="Root of VIVID/VITON/DressCode")
    parser.add_argument("--train_data_meta", type=str, required=True, help="JSONL file path")
    parser.add_argument("--video_sample_size", type=int, default=512, help="Width (Height is usually 384 or adapted)")
    parser.add_argument("--video_sample_n_frames", type=int, default=49)
    parser.add_argument("--video_repeat", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading")
    parser.add_argument("--video_sample_stride", type=int, default=1, help="Stride for video sampling")
    parser.add_argument("--max_res", type=int, default=512, help="Max resolution for training")
    parser.add_argument("--filter_type", type=str, default="image", choices=["image", "video"], help="Filter dataset for image or video samples")

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
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    # Lora 
    parser.add_argument("--rank", type=int, default=128, help="The dimension of the LoRA update matrices")
    parser.add_argument("--network_alpha", type=float, default=64, help="The alpha parameter for LoRA scaling")
    
    #evaluation
    parser.add_argument("--test_data_meta", type=str, default=None, help="Path to validation dataset metadata")
    parser.add_argument("--test_steps", type=int, default=500, help="Run validation every X steps")
    parser.add_argument("--test_height", type=int, default=1024, help="sample test height")
    parser.add_argument("--test_width", type=int, default=768, help="sample test width")
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
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).eval()
    
    # # 3. CLIP Image Encoder (For Cloth condition)
    # logger.info("load clip_encoder!")
    # clip_image_encoder = CLIPModel.from_pretrained(
    #     os.path.join(args.pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    # ).eval()

    # 4. Transformer (Trainable)
    logger.info("load dit!")
    transformer3d = WanTransformer3DTryonModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False    
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
    # clip_image_encoder.requires_grad_(False)
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


    lora_config = LoraConfig(
        r=args.rank, 
        lora_alpha=args.network_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o"
        ],
        modules_to_save=[
            "patch_embedding", 
            "subject_image_proj_model", 
            "k_ip",   
            "v_ip",    
            "norm_k_ip",
            "adapter", 
            "cloth_patch_embedding"
        ]
    )

    transformer3d = get_peft_model(transformer3d, lora_config)
    transformer3d.print_trainable_parameters()
    transformer3d.to(weight_dtype)

    # --- D. Optimizer & Scheduler ---
    optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_count = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in transformer3d.parameters())
    logger.info(f"🔥🔥🔥 Final Trainable Parameters: {trainable_params_count / 1e6:.2f} M / {total_params / 1e6:.2f} M")
    transformer3d.enable_gradient_checkpointing() if args.gradient_checkpointing else None
    optimizer = optimizer_cls(trainable_params, lr=args.learning_rate)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs'])))


    def vton_collate_fn(examples, max_res=512):
        """
        功能：
        1. 空间上：保持原长宽比，长边强制缩放到 max_res，并 32 对齐。
        2. 时间上：以 Batch 中第一个样本的帧数为基准，强制对齐其他样本（复制或裁切）。
        """
        # ---------------------------------------------------
        # 1. 计算目标分辨率 (Spatial)
        # ---------------------------------------------------
        raw_max_h = max([ex["pixel_values"].shape[-2] for ex in examples])
        raw_max_w = max([ex["pixel_values"].shape[-1] for ex in examples])

        current_long_side = max(raw_max_h, raw_max_w)

        # 强制缩放到 max_res
        scale = max_res / current_long_side

        target_h = int(raw_max_h * scale)
        target_w = int(raw_max_w * scale)

        # VAE 32对齐
        target_h = max(32, round(target_h / 32) * 32)
        target_w = max(32, round(target_w / 32) * 32)

        # ---------------------------------------------------
        # 2. 确定目标帧数 (Temporal)
        # ---------------------------------------------------
        # 以第一个样本为基准。
        # 如果是 Image Stage，这里是 1；如果是 Video Stage，这里是 49
        target_len = examples[0]["pixel_values"].shape[0] 

        # ---------------------------------------------------
        # 3. 处理数据
        # ---------------------------------------------------
        aligned_examples = []
        # 需要处理时间维度的 key
        temporal_keys = ["pixel_values", "densepose_pixel_values", "agnostic_pixel_values", "mask_pixel_values"]

        for ex in examples:
            new_ex = {}
            new_ex["data_type"] = ex["data_type"]
            new_ex["text"] = ex["text"]
            
            # 获取当前样本的帧数
            curr_len = ex["pixel_values"].shape[0]

            for key in temporal_keys:
                if key not in ex: continue
                tensor = ex[key] # Shape: [T, C, H, W]
                
                # --- A. 时间维度对齐 (Temporal Align) ---
                if curr_len < target_len:
                    # 如果当前短 (e.g. 1 < 49)，重复填充
                    repeat_times = target_len // curr_len + 1
                    tensor = tensor.repeat(repeat_times, 1, 1, 1)[:target_len]
                elif curr_len > target_len:
                    # 如果当前长 (e.g. 60 > 49)，截取
                    tensor = tensor[:target_len]
                
                # --- B. 空间维度对齐 (Spatial Resize) ---
                mode = 'nearest' if 'mask' in key or 'densepose' in key else 'bilinear'
                align_corners = False if mode != 'nearest' else None
                
                # F.interpolate 接受 [N, C, H, W]，这里 T 维度充当 N，正好批量处理每一帧
                if tensor.shape[-2] != target_h or tensor.shape[-1] != target_w:
                    tensor = F.interpolate(tensor, size=(target_h, target_w), mode=mode, align_corners=align_corners)
                
                new_ex[key] = tensor

            # --- Cloth 单独处理 (始终是单帧图像，不需要时间对齐) ---
            if "cloth_pixel_values" in ex:
                cloth = ex["cloth_pixel_values"] # [1, C, H, W]
                if cloth.shape[-2] != target_h or cloth.shape[-1] != target_w:
                    cloth = F.interpolate(cloth, size=(target_h, target_w), mode='bilinear', align_corners=False)
                new_ex["cloth_pixel_values"] = cloth

            aligned_examples.append(new_ex)

        # 4. Stack
        batch = {
            "pixel_values": torch.stack([ex["pixel_values"] for ex in aligned_examples]),
            "cloth_pixel_values": torch.stack([ex["cloth_pixel_values"] for ex in aligned_examples]),
            "agnostic_pixel_values": torch.stack([ex["agnostic_pixel_values"] for ex in aligned_examples]),
            "mask_pixel_values": torch.stack([ex["mask_pixel_values"] for ex in aligned_examples]),
            "densepose_pixel_values": torch.stack([ex["densepose_pixel_values"] for ex in aligned_examples]),
            "text": [ex["text"] for ex in aligned_examples],
            "data_type": [ex["data_type"] for ex in aligned_examples]
        }

        prompt_ids = tokenizer(
            batch["text"], 
            max_length=512, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        batch['input_ids'] = prompt_ids.input_ids
        batch['attention_mask'] = prompt_ids.attention_mask
        return batch

    # --- E. Dataset & Dataloader (Crucial Change for VTON) ---
    print(f"Initializing VTON Dataset...")
    # 这里直接使用我们自定义的 Dataset，移除了 Bucket 逻辑
    train_dataset = TryOnDataset(
        ann_path=args.train_data_meta,
        data_root=args.data_dir,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        video_repeat=args.video_repeat,
        text_drop_ratio=0.0,
        filter_type=args.filter_type
    )

    base_sampler = RandomSampler(train_dataset)
    
    # 2. 分组采样 (使用新 JSONL 的宽高)
    batch_sampler = AspectRatioBatchImageVideoSamplerTryOn(
        sampler=base_sampler,
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        aspect_ratios=ASPECT_RATIO_512,
        drop_last=False
    )

    collate_fn_with_res = partial(vton_collate_fn, max_res=args.max_res)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn_with_res,
        num_workers=args.dataloader_num_workers,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler , optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps, num_training_steps=args.max_train_steps
    )

    transformer3d, optimizer, train_dataloader, lr_scheduler= accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
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
        
        accelerator.init_trackers("wan_vton_project", config=tracker_config,init_kwargs={"wandb": {"name": args.output_dir}})

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    dino_image_encoder.to(accelerator.device, dtype=weight_dtype)
    siglip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # --- G. Training Loop ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    initial_global_step = 0
    first_epoch = 0
    global_step = 0

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
                    cloth_latents = encode_vae(cloth_values)

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
                    
                    # clip_context = clip_image_encoder(cloth_values)

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
                # clip_context = clip_context.to(weight_dtype)
                inpaint_latents = inpaint_latents.to(weight_dtype)
                cloth_latents = cloth_latents.to(weight_dtype)

                with accelerator.autocast():
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        seq_len=seq_len,
                        context=encoder_hidden_states,
                        t=timesteps,
                        y=inpaint_latents,   
                        # clip_fea=clip_context,
                        cloth_latents=cloth_latents,
                        subject_image_embeds_dict=subject_image_embeds_dict,
                    )

                pixel_mask = batch["mask_pixel_values"].to(device=latents.device, dtype=latents.dtype)

                # 2. 将 Mask 调整为 Latent 尺寸
                # model_pred 的形状通常是 [B, C, F, H_latent, W_latent]
                # 我们只取 spatial 维度 (H_latent, W_latent) 进行插值
                target_h, target_w = noise_pred.shape[-2], noise_pred.shape[-1]

                # 使用 nearest 插值保持二值特性（要么是0要么是1，边缘更锐利）
                # 如果 mask 是 5D [B, C, F, H, W]，interpolate 需要 reshape 或者只对最后两维操作
                if pixel_mask.shape[-2:] != (target_h, target_w):
                    # F.interpolate 对 5D 输入支持有限，通常建议压扁或者只处理 spatial
                    # 简单做法：reshape 成 [B*F, 1, H, W] 处理完再变回来，或者直接用 trilinear (比较慢且边缘模糊)
                    # 这里推荐最稳妥的 spatial resize 方式:
                    b, c, f, h, w = pixel_mask.shape
                    pixel_mask_reshaped = pixel_mask.transpose(1, 2).reshape(b * f, c, h, w) # [B*F, 1, H, W]
                    
                    latent_mask = torch.nn.functional.interpolate(
                        pixel_mask_reshaped, 
                        size=(target_h, target_w), 
                        mode="nearest"  # 推荐 nearest 保持硬边界，bilinear 会产生 0.5 的灰边
                    )
                    
                    # 变回 5D: [B, C, F, H, W]
                    latent_mask = latent_mask.reshape(b, f, c, target_h, target_w).transpose(1, 2)
                else:
                    latent_mask = pixel_mask

                # 3. 广播 Mask 以匹配 Latent 通道数
                # Latent 通常有 16 个通道 (Wan2.1 VAE)，Mask 只有 1 个通道
                # 形状变为 [B, 16, F, H, W] 以便相乘
                # (其实 PyTorch 会自动广播，这步可以省略，但为了逻辑清晰可以写)
                # latent_mask = latent_mask.expand_as(model_pred) 

                # 4. 计算 Loss
                # 先算差的平方 (Element-wise Squared Error)
                diff = noise_pred - target
                loss_map = diff ** 2

                # 乘以 Mask (背景区域 Loss 变 0，Mask 区域 Loss 保留)
                masked_loss_map = loss_map * latent_mask

                # 5. 归一化 (Sum / Count)
                # 注意：分母不能直接除以总像素，要除以 Mask 为 1 的像素数
                # 加上 1e-6 防止除以 0 (虽然训练数据里应该都有 Mask)
                loss = masked_loss_map.sum() / (latent_mask.sum() * noise_pred.shape[1] + 1e-6)

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

                # if global_step % args.test_steps == 0:
                #     if accelerator.is_main_process:
                #         logger.info(f"🔍 Starting validation at step {global_step}...")
                #         transformer3d.eval()
                #         unwrapped_model = accelerator.unwrap_model(transformer3d)
                #         validation_pipeline = WanTryOnPipeline(
                #             tokenizer=tokenizer,
                #             text_encoder=text_encoder,
                #             vae=vae,
                #             transformer=unwrapped_model,  # 直接传入训练中的 PeftModel
                #             siglip_image_encoder=siglip_image_encoder,
                #             siglip_image_processor=siglip_image_processor,
                #             dino_image_encoder=dino_image_encoder,
                #             dino_image_processor=dino_image_processor,
                #             scheduler=noise_scheduler # 使用你训练用的 scheduler 即可
                #         )
                #         validation_pipeline.to(accelerator.device, weight_dtype)
                #         test_dataset = TryOnDataset(
                #             ann_path=args.test_data_meta,
                #             data_root=args.data_dir,
                #             video_sample_stride=args.video_sample_stride,
                #             video_sample_n_frames=args.video_sample_n_frames,
                #             video_repeat=args.video_repeat,
                #             text_drop_ratio=0.0,
                #             filter_type=args.filter_type
                #         )
                #         total_samples = len(test_dataset)
                #         num_samples = 5
                #         selected_indices = random.sample(range(total_samples), min(num_samples, total_samples))

                #         wandb_log_images = []

                #         try:
                #             with torch.no_grad():
                #                 for idx in selected_indices:
                #                     sample = test_dataset[idx]
                #                     base_name = f"step{global_step}_idx{idx}" # 给文件名加个前缀

                #                     # --- 1. 辅助函数：将 Tensor 转为 PIL Image ---
                #                     def tensor_2_pil(t, is_mask=False):
                #                         if t.ndim == 5: t = t[0, :, 0, :, :] 
                #                         elif t.ndim == 4: t = t[:, 0, :, :] 
                #                         elif t.ndim == 3: t = t
                                        
                #                         t = t.float().cpu().permute(1, 2, 0).numpy()
                                        
                #                         # 如果是输入图像(agnostic/cloth)，通常是[-1, 1]归一化的，需要转回[0, 1]
                #                         if not is_mask and t.min() < 0:
                #                             t = (t / 2 + 0.5)
                                        
                #                         t = (t * 255).clip(0, 255).astype(np.uint8)
                #                         if t.shape[2] == 1: t = t.squeeze(2)
                #                         return Image.fromarray(t)

                #                     # --- 2. 辅助函数：将 Output Numpy 转为 PIL Image ---
                #                     def video_2_pil(v):
                #                         if v.ndim == 5: v = v[0]
                #                         if v.ndim == 4: v = v[:, 0, :, :]
                #                         if v.shape[0] in [1, 3]: v = np.transpose(v, (1, 2, 0))
                #                         if v.shape[-1] == 1: v = v.squeeze(-1)
                #                         v = (v * 255).clip(0, 255).astype(np.uint8)
                #                         return Image.fromarray(v)

                #                     # --- 3. 准备 Tensor (和你原来的逻辑一样) ---
                #                     def prepare_tensor(t, is_mask=False):
                #                         t = t.to(accelerator.device, weight_dtype) # 修正：直接用 accelerator.device
                #                         if t.ndim == 3: t = t.unsqueeze(0)
                #                         if t.ndim == 4: t = t.unsqueeze(0)
                #                         t = t.permute(0, 2, 1, 3, 4)
                #                         # ... (你原来的 Resize 逻辑保留) ...
                #                         if t.shape[-2] != args.test_height or t.shape[-1] != args.test_width:
                #                              mode = 'nearest' if is_mask else 'bilinear'
                #                              b, c, f, h, w = t.shape
                #                              t_flattened = t.transpose(1, 2).reshape(b * f, c, h, w)
                #                              t_resized = torch.nn.functional.interpolate(t_flattened, size=(args.test_height, args.test_width), mode=mode)
                #                              t = t_resized.reshape(b, f, c, args.test_height, args.test_width).transpose(1, 2)
                #                         return t

                #                     pixel_t  = prepare_tensor(sample["pixel_values"])
                #                     agnostic_t = prepare_tensor(sample["agnostic_pixel_values"])
                #                     mask_t = prepare_tensor(sample["mask_pixel_values"], is_mask=True)
                #                     densepose_t = prepare_tensor(sample["densepose_pixel_values"])
                #                     cloth_t = prepare_tensor(sample["cloth_pixel_values"])
                #                     prompt = sample["text"]

                #                     # --- 4. 推理 ---
                #                     output = validation_pipeline(
                #                         prompt=prompt,
                #                         image=agnostic_t,
                #                         mask_image=mask_t,
                #                         densepose_image=densepose_t,
                #                         cloth_image=cloth_t,
                #                         num_frames=1,
                #                         height=args.test_height,
                #                         width=args.test_width,
                #                         num_inference_steps=20,
                #                         guidance_scale=5.0,
                #                         output_type="numpy"
                #                     )

                #                     # --- 5. 转换并保存到本地 ---
                #                     pil_agnostic = tensor_2_pil(agnostic_t)
                #                     pil_cloth = tensor_2_pil(cloth_t)
                #                     pil_pose = tensor_2_pil(densepose_t)
                #                     pil_pixel = tensor_2_pil(pixel_t)
                #                     pil_result = video_2_pil(output.videos)

                #                     # --- 6. 【关键】拼接图片用于 WandB 展示 ---
                #                     # 顺序：模特原图(去掉了衣服) | 衣服图 | Densepose | 生成结果
                #                     # 创建一张宽画布
                #                     w, h = pil_result.size
                #                     combo_image = Image.new('RGB', (w * 5, h))
                #                     combo_image.paste(pil_agnostic, (0, 0))
                #                     combo_image.paste(pil_cloth, (w, 0))
                #                     combo_image.paste(pil_pose, (w * 2, 0))
                #                     combo_image.paste(pil_result, (w * 3, 0))
                #                     combo_image.paste(pil_pixel, (w*4, 0))

                #                     # 添加到列表，准备上传
                #                     # caption 可以写 prompt 或者 index
                #                     wandb_log_images.append(
                #                         wandb.Image(combo_image, caption=f"Step {global_step} | Idx {idx}")
                #                     )

                #                 logger.info("Validation inference completed.")
                                
                #                 # --- 7. 一次性上传到 WandB ---
                #                 if len(wandb_log_images) > 0:
                #                     # 检查 wandb tracker 是否存在
                #                     tracker = accelerator.get_tracker("wandb")
                #                     if tracker is not None:
                #                         tracker.log(
                #                             {"validation_samples": wandb_log_images}, 
                #                             step=global_step
                #                         )
                #                     else:
                #                         if wandb.run is not None:
                #                             wandb.log({"validation_samples": wandb_log_images}, step=global_step)

                #         except Exception as e:
                #             logger.error(f"Validation failed: {e}")
                #             import traceback
                #             traceback.print_exc()

                #         # 5. 清理现场 (这一步至关重要！)
                #         del validation_pipeline
                #         torch.cuda.empty_cache()
                #         transformer3d.train()

                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped_model = accelerator.unwrap_model(transformer3d)
                    unwrapped_model.save_pretrained(save_path)
                    logger.info(f"Saved PEFT inference weights to {save_path}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved full training state (optimizer/scheduler) to {save_path}")

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()

if __name__ == "__main__":
    main()