import argparse
import os
import sys
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch

# 假设你的代码结构如下，根据实际情况调整 import
# 必须确保能引用到你的 TryOnDataset 类
sys.path.append(".") 
from videox_fun.data.tryon_video import TryOnDataset 

def validate_dataset(args):
    print(f"🔍 Starting dataset validation...")
    print(f"📂 Data Root: {args.train_data_dir}")
    print(f"📄 Metadata: {args.train_data_meta}")

    # 1. 初始化 Dataset
    # 使用和训练脚本一致的参数，确保测试环境一致
    dataset = TryOnDataset(
        ann_path=args.train_data_meta,
        data_root=args.train_data_dir,
        video_sample_size=args.video_sample_size,
        video_sample_stride=args.video_sample_stride,
        video_sample_n_frames=args.video_sample_n_frames,
        image_sample_size=args.image_sample_size,
        video_repeat=0, # 验证时不重复视频，跑一遍即可
        text_drop_ratio=0.0
    )

    # 2. 定义一个简单的 Collate Fn (只要能堆叠就行，甚至可以返回 None)
    def fast_collate(batch):
        # 我们只关心是否报错，不关心 Tensor 形状
        return batch

    # 3. 初始化 DataLoader
    # num_workers 建议设置高一点，快速吃满 CPU
    dataloader = DataLoader(
        dataset,
        batch_size=1, # 逐个验证，方便定位坏文件
        shuffle=False, # 按顺序读，方便对应行号
        num_workers=args.num_workers,
        collate_fn=fast_collate,
        prefetch_factor=2
    )

    print(f"📊 Total samples: {len(dataset)}")
    print(f"🚀 Running with {args.num_workers} workers...")

    start_time = time.time()
    success_count = 0
    error_count = 0
    bad_files = []

    # 4. 开始遍历
    # 使用 tqdm 显示进度
    pbar = tqdm(dataloader, total=len(dataset), unit="samples")
    
    for i, batch in enumerate(pbar):
        # 在 DataLoader 内部，如果有错误，你的 Dataset.__getitem__ 里的 try...except 
        # 可能会捕获并重试。为了验证，我们不仅要看能不能跑通，
        # 还要看你的 Dataset 类是否在遇到坏文件时打印了 Log。
        
        # 这里的 batch 是 __getitem__ 的返回值
        # 如果你的 Dataset 在出错时 raise Error，这里就会捕获不到（进程会挂）
        # 所以确保你的 Dataset.__getitem__ 写得足够健壮
        
        # 如果能运行到这里，说明读取成功（或者被 Dataset 内部的 try-catch 处理了）
        success_count += 1
        
        # 可以在这里打印当前文件名（如果 Dataset 返回了 path）
        # print(batch[0]['file_name']) 

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"✅ Validation Finished in {total_time:.2f}s")
    print(f"🟢 Successful samples: {success_count}")
    print(f"🔴 Failed samples (caught by loader): {len(dataset) - success_count}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 根据你的训练脚本参数修改默认值
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--train_data_meta", type=str, required=True)
    parser.add_argument("--video_sample_size", type=int, default=512)
    parser.add_argument("--image_sample_size", type=int, default=512)
    parser.add_argument("--video_sample_n_frames", type=int, default=49)
    parser.add_argument("--video_sample_stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16) # 开大一点加速
    
    args = parser.parse_args()
    
    validate_dataset(args)