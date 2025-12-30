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
# resize_frame 如果不需要在这里缩放就不引用了，或者保留用来做基础降采样
from func_timeout import func_timeout

class TryOnDataset(Dataset):
    def __init__(
        self,
        ann_path, 
        data_root, 
        # 下面这些尺寸参数在 Dataset 里暂时没用了，但保留接口不报错
        video_sample_size=512, 
        image_sample_size=512,
        video_sample_stride=1, 
        video_sample_n_frames=49, 
        video_repeat=0,
        text_drop_ratio=0.0, 
        enable_bucket=False, 
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
    ):
        print(f"loading annotations from {ann_path} ...")
        self.data_root = data_root
        
        # 1. Load Metadata
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json') or ann_path.endswith('.jsonl'):
            dataset = []
            with open(ann_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line))
        
        # 2. Data Balancing
        if video_repeat > 0:
            self.dataset = []
            video_data = [d for d in dataset if d.get('type') == 'video']
            image_data = [d for d in dataset if d.get('type') != 'video']
            self.dataset.extend(image_data)
            for _ in range(video_repeat):
                self.dataset.extend(video_data)
        else:
            self.dataset = dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        # Params
        self.video_sample_stride = video_sample_stride
        self.video_sample_n_frames = video_sample_n_frames
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.text_drop_ratio = text_drop_ratio
        
        # Transforms
        # 只保留数值归一化，不包含任何几何变换 (Resize/Crop)
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

    def _get_dataset_subpath(self, data_info):
        # ... (保持原有路径解析逻辑不变) ...
        data_type = data_info.get('type', 'image')
        category = data_info.get('category', '')
        
        if data_type == 'video':
            dataset_name = 'vivid'
        elif 'viton' in category:
            dataset_name = 'viton_hd'
        else:
            dataset_name = 'dresscode'
            
        filename = data_info['x']
        train_path = os.path.join(self.data_root, dataset_name, 'train', 'image', filename)
        if os.path.exists(train_path) or os.path.exists(train_path.replace('.jpg', '.mp4')):
            split = 'train'
        else:
            split = 'test'
            
        return dataset_name, split

    def _resolve_paths(self, data_info):
        # ... (保持原有路径检查逻辑不变) ...
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
        只读取，不缩放，不裁切
        """
        img = Image.open(img_path).convert('RGB')
        if is_mask:
            img = img.convert('L')
            tensor = self.mask_transform(img)
        else:
            tensor = self.norm_transform(img)
            
        # 返回 shape: [C, H, W] (增加的 batch 维由 collate_fn 负责，这里如果不 unsqueeze 也可以，看你习惯)
        return tensor.unsqueeze(0) 

    def _process_video_frames(self, frames_list, is_mask=False):
        """
        frames_list: list of numpy arrays (H, W, C)
        直接转换为 Tensor，保留原始分辨率
        """
        # 直接堆叠
        np_frames = np.array(frames_list) # shape: (T, H, W, C)
        
        if is_mask:
            # Mask 处理: (T, H, W, 1) -> Permute -> Normalize
            tensor = torch.from_numpy(np_frames).float() / 255.0
            tensor = tensor.permute(0, 3, 1, 2) # (T, C, H, W)
            if tensor.shape[1] == 3:
                 tensor = tensor[:, 0:1, :, :]
        else:
            # RGB 处理: (T, H, W, 3) -> Permute -> Normalize (-1, 1)
            tensor = torch.from_numpy(np_frames).permute(0, 3, 1, 2).float() # (T, C, H, W)
            tensor = (tensor / 255.0 - 0.5) / 0.5
            
        # ================= 修改处 =================
        # 原来的新代码里有这一行，把它【删掉】或【注释掉】
        # tensor = tensor.permute(1, 0, 2, 3) # 不要这行，保持 (T, C, H, W)
        # =========================================
            
        return tensor

    def get_batch(self, idx):
        # 这里的逻辑和之前一样，只是调用的 _process 函数变了
        data_info = self.dataset[idx % len(self.dataset)]
        paths = self._resolve_paths(data_info)
        
        if data_info.get('type', 'image') != 'video':
            # Image Mode
            sample = {}
            sample['input'] = self._process_image(paths['input'])
            sample['cloth'] = self._process_image(paths['cloth'])
            sample['densepose'] = self._process_image(paths['densepose'])
            sample['agnostic'] = self._process_image(paths['agnostic'])
            sample['agnostic_mask'] = self._process_image(paths['agnostic_mask'], is_mask=True)
            
            text = data_info.get('caption', '')
            if random.random() < self.text_drop_ratio: text = ''
            
            return sample, text, 'image', paths['input']

        else:
            # Video Mode
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
            # 注意：这里返回的 Tensor 尺寸取决于原视频分辨率，可能每个 Batch 都不一样
            sample['input'] = self._process_video_frames(input_frames)
            sample['densepose'] = self._process_video_frames(video_components['densepose'])
            sample['agnostic'] = self._process_video_frames(video_components['agnostic'])
            sample['agnostic_mask'] = self._process_video_frames(video_components['agnostic_mask'], is_mask=True)
            sample['cloth'] = self._process_image(paths['cloth']) # 衣服通常是单张图

            text = data_info.get('caption', '')
            if random.random() < self.text_drop_ratio: text = ''
            
            return sample, text, 'video', paths['input']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 错误重试逻辑
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts:
            try:
                sample_dict, text, dtype, fpath = self.get_batch(idx)
                
                ret = {
                    "pixel_values": sample_dict['input'],
                    "cloth_pixel_values": sample_dict['cloth'],
                    "densepose_pixel_values": sample_dict['densepose'],
                    "agnostic_pixel_values": sample_dict['agnostic'],
                    "mask_pixel_values": sample_dict['agnostic_mask'],
                    "text": text,
                    "data_type": dtype,
                    "file_name": os.path.basename(fpath),
                    # 新增：把原始形状传出去，方便 Collate 决定怎么 Bucket
                    "raw_shape": sample_dict['input'].shape # (C, T, H, W)
                }
                return ret
                
            except Exception as e:
                print(f"[Dataset Error] Failed to load idx {idx}. Attempt {attempts+1}. Error: {e}")
                idx = random.randint(0, self.length - 1)
                attempts += 1
        
        raise RuntimeError(f"Failed to load valid batch after {max_attempts} attempts.")