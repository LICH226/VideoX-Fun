#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import sys
import random
import torch
import torch.nn.functional as F
import numpy as np
from functools import partial
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import RandomSampler, DataLoader
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 路径 Hack (确保能 import videox_fun)
# ---------------------------------------------------------------------------
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

# 引入你的 Dataset 和 Sampler
from videox_fun.data.tryon_video import TryOnDataset
from videox_fun.data.bucket_sampler import AspectRatioBatchImageVideoSamplerTryOn, ASPECT_RATIO_512

# ---------------------------------------------------------------------------
# 1. 新版 Collate Fn (复制过来，或者从 train.py import)import torch
import torch.nn.functional as F

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
    
    return batch

# ---------------------------------------------------------------------------
# 2. Main Check Logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Main Check Logic
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--train_data_meta", type=str, required=True)
    
    parser.add_argument("--filter_type", type=str, default="image", choices=["image", "video"])
    parser.add_argument("--max_res", type=int, default=512)
    
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_check_steps", type=int, default=50, help="打印前50个batch看看")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator()
    if args.seed: set_seed(args.seed)

    if accelerator.is_main_process:
        print(f"\n🚀 [Check Start] Mode: {args.filter_type.upper()} | Target Max Res: {args.max_res}")

    # 1. 初始化 Dataset
    dataset = TryOnDataset(
        ann_path=args.train_data_meta,
        data_root=args.train_data_dir,
        # 这些参数其实都不重要了，因为都在 Collate 里处理，但为了兼容性传进去
        video_sample_n_frames=49, 
        filter_type=args.filter_type 
    )

    # 2. Sampler
    batch_sampler = AspectRatioBatchImageVideoSamplerTryOn(
        sampler=RandomSampler(dataset),
        dataset=dataset,
        batch_size=args.train_batch_size,
        aspect_ratios=ASPECT_RATIO_512,
        drop_last=True
    )

    # 3. Collate (绑定 max_res)
    collate_fn = partial(vton_collate_fn, max_res=args.max_res)

    # 4. DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        collate_fn=collate_fn,
        num_workers=4
    )
    dataloader = accelerator.prepare(dataloader)

    # 5. Check Loop
    iterator = iter(dataloader)
    
    # 使用 tqdm，但在循环内部强制 print
    for step in range(args.max_check_steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        # 获取当前卡信息
        local_tensor = batch['pixel_values']
        local_type = batch['data_type'][0]
        
        B, T, C, H, W = local_tensor.shape
        type_flag = 1.0 if local_type == 'video' else 0.0
        
        # 构造信息向量: [B, T, H, W, Type]
        info_vec = torch.tensor([float(B), float(T), float(H), float(W), type_flag], device=accelerator.device).unsqueeze(0)
        
        # 集合通信 Gather
        gathered = accelerator.gather(info_vec) # [Num_GPU, 5]

        if accelerator.is_main_process:
            first = gathered[0]
            
            # --- 校验一致性 ---
            is_valid = True
            for i in range(1, gathered.shape[0]):
                if not torch.equal(first, gathered[i]):
                    is_valid = False
                    print(f"❌ [FAIL] Step {step}: Mismatch GPU 0 vs GPU {i}")
                    print(f"   GPU 0: {first.tolist()}")
                    print(f"   GPU {i}: {gathered[i].tolist()}")
                    break
            
            # --- 打印形状 ---
            if is_valid:
                b_val = int(first[0].item())
                t_val = int(first[1].item())
                h_val = int(first[2].item())
                w_val = int(first[3].item())
                
                # 打印格式: Step | [B, C, T, H, W]
                print(f"Step {step:03d} | Shape: [{b_val}, 3, {t_val}, {h_val}, {w_val}] | Res: {h_val}x{w_val}")

    print("🎉 Check Finished.")

if __name__ == "__main__":
    main()