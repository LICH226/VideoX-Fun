#!/usr/bin/env python
# coding=utf-8

import argparse
import os
import torch
import logging
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm

# 引用你的项目模块
from videox_fun.data.tryon_video import TryOnDataset 
from videox_fun.data.dataset_image_video import ImageVideoSampler 

# 设置简单的日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="DataLoader Check Script")
    
    # 只保留数据相关的参数
    parser.add_argument("--train_data_dir", type=str, required=True, help="Root of VIVID/VITON/DressCode")
    parser.add_argument("--train_data_meta", type=str, required=True, help="JSONL file path")
    parser.add_argument("--video_sample_size", type=int, default=512)
    parser.add_argument("--video_sample_n_frames", type=int, default=49)
    parser.add_argument("--video_repeat", type=int, default=0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4) # 稍微给点 worker
    parser.add_argument("--video_sample_stride", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # 1. 初始化 Accelerator
    accelerator = Accelerator()
    
    # 打印当前进程信息
    logger.info(f"🚀 Process starting: Rank {accelerator.process_index} / {accelerator.num_processes}")
    
    if args.seed is not None:
        set_seed(args.seed) # 注意：这里设定了相同的 Seed，会导致 RandomSampler 在所有卡上行为一致

    # 2. 初始化 Dataset
    if accelerator.is_local_main_process:
        print(f"Initializing Dataset...")
        
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

    # 3. Collate Function (保持原样以防报错)
    def vton_collate_fn(examples):
        # 简化版 collate，只提取我们需要验证的字段，避免 tensor 计算消耗时间
        # 我们主要看 text 字段来区分样本
        texts = [ex["text"] for ex in examples]
        
        # 为了让 DataLoader 跑通，返回一个简单的 dict
        # 实际训练中这里会有复杂的 tensor stack，这里省略以加速
        return {"text": texts}

    # 4. DataLoader (完全复刻你当前的代码逻辑)
    # 注意：这里使用了 ImageVideoSampler 包裹 RandomSampler
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=ImageVideoSampler(RandomSampler(train_dataset), train_dataset, args.train_batch_size),
        collate_fn=vton_collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # 5. Accelerator Prepare
    #这是关键：看看 prepare 之后，accelerator 是否能修正你的 custom batch_sampler
    train_dataloader = accelerator.prepare(train_dataloader)

    # 6. 验证循环
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("  STARTING DATALOADER CHECK LOOP")
        print("="*50 + "\n")

    # 只跑前 5 个 step
    check_steps = 5
    
    for step, batch in enumerate(train_dataloader):
        if step >= check_steps:
            break
            
        current_texts = batch['text']
        
        # 格式化输出，方便 grep
        # 格式: [CHECK] Step: X | Rank: Y | Data: ...
        for txt in current_texts:
            print(f"[CHECK] Step: {step} | Rank: {accelerator.process_index} | Data: {txt}")
            
        # 稍微同步一下，让打印不那么乱（虽然还是会乱，但好一点）
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("\n" + "="*50)
        print("  CHECK FINISHED")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()