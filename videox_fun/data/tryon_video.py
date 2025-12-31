import os
import json
import csv
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import VideoReader_contextmanager, get_video_reader_batch, VIDEO_READER_TIMEOUT
from func_timeout import func_timeout

class TryOnDataset(Dataset):
    def __init__(
        self,
        ann_path, 
        data_root, 
        video_sample_n_frames=49, 
        video_sample_stride=1,
        text_drop_ratio=0.0, 
        enable_bucket=False, # 已废弃
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        filter_type="all", # "image", "video", "all"
        **kwargs # 吸收多余参数
    ):
        print(f"loading annotations from {ann_path} ...")
        self.data_root = data_root
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.text_drop_ratio = text_drop_ratio
        
        # 1. Load Metadata
        raw_dataset = []
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                raw_dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json') or ann_path.endswith('.jsonl'):
            with open(ann_path, 'r') as f:
                for line in f:
                    raw_dataset.append(json.loads(line))
        
        # 2. Filter Data (分阶段训练核心：只加载特定类型)
        self.dataset = []
        if filter_type == 'image':
            print("🚀 Stage 1: Loading IMAGE data only...")
            self.dataset = [d for d in raw_dataset if d.get('type') != 'video']
        elif filter_type == 'video':
            print("🚀 Stage 2: Loading VIDEO data only...")
            self.dataset = [d for d in raw_dataset if d.get('type') == 'video']
        else:
            print("⚠️ Loading ALL data (Mixed training). Make sure your strategy supports this.")
            self.dataset = raw_dataset

        self.length = len(self.dataset)
        print(f"✅ Final Data Scale: {self.length}")

        # Transforms: 只做归一化，不做 Resize/Crop
        # 我们希望把原图/原视频扔给 Collate_fn，让它根据当前 Batch 情况去决定缩放，最大程度保留清晰度
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

    def _get_dataset_subpath(self, data_info):
        data_type = data_info.get('type', 'image')
        category = data_info.get('category', '')
        
        if data_type == 'video':
            dataset_name = 'vivid'
        elif 'viton' in category:
            dataset_name = 'viton_hd'
        else:
            dataset_name = 'dresscode'
            
        filename = data_info['x']
        # 简化 split 判断逻辑，假设都存在 train 文件夹下
        # 实际逻辑应根据文件是否存在判断，保持你原来的逻辑即可
        train_path = os.path.join(self.data_root, dataset_name, 'train', 'image', filename)
        if os.path.exists(train_path) or os.path.exists(train_path.replace('.jpg', '.mp4')):
            split = 'train'
        else:
            split = 'test'
            
        return dataset_name, split

    def _resolve_paths(self, data_info):
        dataset_name, split = self._get_dataset_subpath(data_info)
        base_dir = os.path.join(self.data_root, dataset_name, split)
        
        filename_x = data_info['x']
        filename_cloth = data_info['cloth']
        
        paths = {
            "input": os.path.join(base_dir, "image", filename_x),
            "cloth": os.path.join(base_dir, "cloth", filename_cloth),
            "densepose": os.path.join(base_dir, "densepose", filename_x), 
            "agnostic": os.path.join(base_dir, "agnostic", filename_x),
            "agnostic_mask": os.path.join(base_dir, "agnostic_mask", filename_x),
        }

        check_keys = ['densepose', 'agnostic', 'agnostic_mask']
        for key in check_keys:
            if not os.path.exists(paths[key]):
                candidate_png = paths[key].replace('.jpg', '.png')
                if os.path.exists(candidate_png):
                    paths[key] = candidate_png
                    continue
                if key == 'agnostic_mask':
                    candidate_mask_png = paths[key].replace('.jpg', '_mask.png')
                    if os.path.exists(candidate_mask_png):
                        paths[key] = candidate_mask_png
                        continue
                    candidate_mask_jpg = paths[key].replace('.jpg', '_mask.jpg')
                    if os.path.exists(candidate_mask_jpg):
                        paths[key] = candidate_mask_jpg
                        continue

        for k, v in paths.items():
            if not os.path.exists(v):
                raise FileNotFoundError(f"Required file for {k} not found: {v}")

        return paths

    def _process_image(self, img_path, is_mask=False):
        """ 
        读取 -> 归一化 -> [1, C, H, W] 
        保留原始 H, W，不做缩放 
        """
        img = Image.open(img_path).convert('RGB')
        if is_mask:
            img = img.convert('L')
            tensor = self.mask_transform(img) # [1, H_raw, W_raw]
        else:
            tensor = self.norm_transform(img) # [3, H_raw, W_raw]
        
        # 增加时间维度
        return tensor.unsqueeze(0) 

    def _process_video_frames(self, frames_list, is_mask=False):
        """ 
        Video: [T, H, W, C] -> [T, C, H, W] 
        保留原始 H, W
        """
        np_frames = np.array(frames_list) # (T, H, W, C)
        
        if is_mask:
            # Mask: 0~1 float
            tensor = torch.from_numpy(np_frames).float() / 255.0
            tensor = tensor.permute(0, 3, 1, 2) # T, C, H, W
            if tensor.shape[1] == 3: tensor = tensor[:, 0:1, :, :]
        else:
            # RGB: -1~1 float
            tensor = torch.from_numpy(np_frames).permute(0, 3, 1, 2).float()
            tensor = (tensor / 255.0 - 0.5) / 0.5
            
        return tensor

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        paths = self._resolve_paths(data_info)
        
        # 1. Image Mode
        if data_info.get('type', 'image') != 'video':
            sample = {}
            sample['input'] = self._process_image(paths['input'])
            sample['cloth'] = self._process_image(paths['cloth'])
            sample['densepose'] = self._process_image(paths['densepose'])
            sample['agnostic'] = self._process_image(paths['agnostic'])
            sample['agnostic_mask'] = self._process_image(paths['agnostic_mask'], is_mask=True)
            
            text = data_info.get('caption', '')
            if random.random() < self.text_drop_ratio: text = ''
            
            # 返回: 原始数据, 文本, 类型, 路径
            return sample, text, 'image', paths['input']

        # 2. Video Mode
        else:
            # 保留原有的 VideoReader 采样逻辑
            with VideoReader_contextmanager(paths['input'], num_threads=2) as vr:
                total_frames = len(vr)
                clip_span = (self.video_sample_n_frames - 1) * self.video_sample_stride + 1
                
                start_frame_idx = int(self.video_length_drop_start * total_frames)
                end_frame_idx = int(self.video_length_drop_end * total_frames)
                valid_len = end_frame_idx - start_frame_idx
                
                if valid_len < clip_span:
                    sample_start_idx = start_frame_idx
                else:
                    sample_start_idx = random.randint(start_frame_idx, end_frame_idx - clip_span)
                
                batch_index = np.linspace(sample_start_idx, sample_start_idx + clip_span - 1, self.video_sample_n_frames, dtype=int)
                batch_index = [i % total_frames for i in batch_index]

                try:
                    input_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(vr, batch_index))
                except Exception as e:
                    raise ValueError(f"Read input video timeout/error: {paths['input']} - {e}")

            video_components = {}
            for key in ['densepose', 'agnostic', 'agnostic_mask']:
                p = paths[key]
                try:
                    with VideoReader_contextmanager(p, num_threads=2) as vr_comp:
                        comp_len = len(vr_comp)
                        safe_index = [i % comp_len for i in batch_index]
                        frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(vr_comp, safe_index))
                        video_components[key] = frames
                except Exception as e:
                    raise ValueError(f"Failed to read {key} video: {p}. Error: {e}")

            sample = {}
            # 这里调用 _process_video_frames，不再缩放，只转 Tensor
            sample['input'] = self._process_video_frames(input_frames)
            sample['densepose'] = self._process_video_frames(video_components['densepose'])
            sample['agnostic'] = self._process_video_frames(video_components['agnostic'])
            sample['agnostic_mask'] = self._process_video_frames(video_components['agnostic_mask'], is_mask=True)
            sample['cloth'] = self._process_image(paths['cloth']) # 衣服始终是单张图

            text = data_info.get('caption', '')
            if random.random() < self.text_drop_ratio: text = ''
            
            return sample, text, 'video', paths['input']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            try:
                sample_dict, text, dtype, fpath = self.get_batch(idx)
                
                ret = {
                    "pixel_values": sample_dict['input'],          # [T, C, H, W]
                    "cloth_pixel_values": sample_dict['cloth'],    # [1, C, H, W]
                    "densepose_pixel_values": sample_dict['densepose'],
                    "agnostic_pixel_values": sample_dict['agnostic'],
                    "mask_pixel_values": sample_dict['agnostic_mask'],
                    "text": text,
                    "data_type": dtype,
                    "file_name": os.path.basename(fpath),
                    # 将原始尺寸传出去，Sampler 已经用过它了，但 CollateFn 计算缩放比例时可能需要
                    # (虽然 CollateFn 可以直接 measure tensor shape)
                }
                return ret
                
            except Exception as e:
                print(f"[Dataset Error] Failed to load idx {idx}. Attempt {attempts+1}. Error: {e}")
                idx = random.randint(0, self.length - 1)
                attempts += 1
        
        raise RuntimeError(f"Failed to load valid batch after {max_attempts} attempts.")