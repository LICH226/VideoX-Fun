import os
import sys
import json
import csv
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from decord import VideoReader
from func_timeout import func_timeout, FunctionTimedOut
from .utils import VideoReader_contextmanager, get_video_reader_batch, resize_frame, VIDEO_READER_TIMEOUT

class TryOnDataset(Dataset):
    def __init__(
        self,
        ann_path, 
        data_root,  # 必须提供，指向包含 vivid, viton_hd, dresscode 的根目录
        video_sample_size=512, 
        video_sample_stride=1,     # 关键：设置为1以保证连续
        video_sample_n_frames=49,  # 关键：设置为49适配VAE
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.0,       # VTON任务通常不需要丢弃caption，除非做无条件生成
        enable_bucket=False,       # 暂不支持 bucket，保持简单
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
    ):
        print(f"loading annotations from {ann_path} ...")
        self.data_root = data_root
        
        # 1. 加载 Metadata
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json') or ann_path.endswith('.jsonl'):
            dataset = []
            with open(ann_path, 'r') as f:
                for line in f:
                    dataset.append(json.loads(line))
        
        # 2. 简单的数据平衡逻辑
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
        
        # 尺寸设置
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.larger_side = max(min(self.image_sample_size), min(self.video_sample_size))

        # Transforms
        # 通用的 Normalize (-1, 1)
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # Mask 不需要 Normalize 到 -1,1，通常保持 0-1 或 单通道
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), # [0, 1]
        ])

    def _get_dataset_subpath(self, data_info):
        """
        根据 metadata 判断数据位于 vivid, viton_hd 还是 dresscode，以及是 train 还是 test
        """
        data_type = data_info.get('type', 'image')
        category = data_info.get('category', '')
        
        # 1. 确定数据集名称
        if data_type == 'video':
            dataset_name = 'vivid'
        elif 'viton' in category:
            dataset_name = 'viton_hd'
        else:
            dataset_name = 'dresscode' # 默认为 dresscode
            
        # 2. 确定 split (train/test)
        # 简单策略：检查文件是否存在于 train 中，如果不在则假设在 test 中
        # 注意：这里需要根据 filename (key 'x') 来判断
        filename = data_info['x']
        
        # 构造尝试路径
        train_path = os.path.join(self.data_root, dataset_name, 'train', 'image', filename)
        if os.path.exists(train_path) or os.path.exists(train_path.replace('.jpg', '.mp4')):
            split = 'train'
        else:
            split = 'test'
            
        return dataset_name, split

    def _resolve_paths(self, data_info):
        """
        生成所有需要的组件的文件路径，自动处理 .jpg/.png 后缀不一致问题
        """
        dataset_name, split = self._get_dataset_subpath(data_info)
        base_dir = os.path.join(self.data_root, dataset_name, split)
        
        filename_x = data_info['x']      # e.g., 047723_0.jpg
        filename_cloth = data_info['cloth']
        
        # 1. 定义基础路径（默认使用 jsonl 里的后缀，通常是 .jpg）
        paths = {
            "input": os.path.join(base_dir, "image", filename_x),
            "cloth": os.path.join(base_dir, "cloth", filename_cloth),
            "densepose": os.path.join(base_dir, "densepose", filename_x), 
            "agnostic": os.path.join(base_dir, "agnostic", filename_x),
            "agnostic_mask": os.path.join(base_dir, "agnostic_mask", filename_x),
        }

        # 2. 智能后缀修正逻辑
        # 针对 Densepose, Agnostic, Mask，如果 jpg 不存在，尝试替换为 png
        # 这是为了解决 DressCode 数据集中 image 是 jpg 但标注是 png 的问题
        check_keys = ['densepose', 'agnostic', 'agnostic_mask']
        
        for key in check_keys:
            if not os.path.exists(paths[key]):
                # 尝试 A: 直接替换后缀 .jpg -> .png
                candidate_png = paths[key].replace('.jpg', '.png')
                if os.path.exists(candidate_png):
                    paths[key] = candidate_png
                    continue # 找到了就跳过后续检查
                
                # 尝试 B: 处理 Mask 的特殊命名 (仅针对 agnostic_mask)
                # 有些数据集 mask 叫 _mask.png 或 _mask.jpg
                if key == 'agnostic_mask':
                    # 尝试 filename_mask.png
                    candidate_mask_png = paths[key].replace('.jpg', '_mask.png')
                    if os.path.exists(candidate_mask_png):
                        paths[key] = candidate_mask_png
                        continue
                    
                    # 尝试 filename_mask.jpg
                    candidate_mask_jpg = paths[key].replace('.jpg', '_mask.jpg')
                    if os.path.exists(candidate_mask_jpg):
                        paths[key] = candidate_mask_jpg

        return paths

    def _process_image(self, img_path, is_mask=False):
        """读取并处理单张图片"""
        img = Image.open(img_path).convert('RGB') # Mask如果是单通道可以 convert('L')
        if is_mask:
            img = img.convert('L') # 强制转单通道
            
        # Resize & Crop
        img = transforms.Resize(min(self.image_sample_size))(img)
        img = transforms.CenterCrop(self.image_sample_size)(img)
        
        if is_mask:
            tensor = self.mask_transform(img) # [1, H, W] in 0-1
        else:
            tensor = self.norm_transform(img) # [3, H, W] in -1, 1
            
        return tensor.unsqueeze(0) # [1, C, H, W] 模拟 T=1

    def _process_video_frames(self, frames_list, is_mask=False):
        """将 numpy frames list 转换为 Tensor"""
        resized_frames = []
        for frame in frames_list:
            # resize_frame 是 utils 里的函数，保持长宽比缩放
            rf = resize_frame(frame, self.larger_side) 
            resized_frames.append(rf)
        
        # Center Crop 手动实现 (针对 numpy array)
        np_frames = np.array(resized_frames) # [T, H, W, C]
        h, w = np_frames.shape[1:3]
        th, tw = self.video_sample_size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        cropped_frames = np_frames[:, i:i+th, j:j+tw, :]
        
        # To Tensor
        if is_mask:
            # Mask通常是单通道，如果是读取RGB视频作为Mask，取第一通道或转灰度
            # 假设 input mask video 是 RGB，且黑白分明
            tensor = torch.from_numpy(cropped_frames).float() / 255.0 # [T, H, W, C] 0-1
            tensor = tensor.permute(0, 3, 1, 2) # [T, C, H, W]
            tensor = tensor[:, 0:1, :, :] # 取单通道 [T, 1, H, W]
        else:
            tensor = torch.from_numpy(cropped_frames).permute(0, 3, 1, 2).float() # [T, C, H, W]
            tensor = (tensor / 255.0 - 0.5) / 0.5 # Normalize to -1, 1
            
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
            
            # Caption
            text = data_info.get('caption', '')
            if random.random() < self.text_drop_ratio: text = ''
            
            return sample, text, 'image', paths['input']

        # 2. Video Mode (VIVID)
        else:
            # 2.1 确定采样索引 (Master Clock)
            # 我们只打开 Input Video 来确定时间轴，其他视频组件必须跟随这个时间轴
            with VideoReader_contextmanager(paths['input'], num_threads=2) as vr:
                total_frames = len(vr)
                # 使用我们之前讨论的连续采样逻辑
                clip_span = (self.video_sample_n_frames - 1) * self.video_sample_stride + 1
                
                start_frame_idx = int(self.video_length_drop_start * total_frames)
                end_frame_idx = int(self.video_length_drop_end * total_frames)
                valid_len = end_frame_idx - start_frame_idx
                
                if valid_len < clip_span:
                    sample_start_idx = start_frame_idx
                else:
                    sample_start_idx = random.randint(start_frame_idx, end_frame_idx - clip_span)
                
                batch_index = np.linspace(sample_start_idx, sample_start_idx + clip_span - 1, self.video_sample_n_frames, dtype=int)
                batch_index = [i % total_frames for i in batch_index] # Loop Padding

                # 读取 Input
                try:
                    input_frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(vr, batch_index))
                except:
                    raise ValueError(f"Read input video timeout: {paths['input']}")

            # 2.2 读取其他 Video 组件 (Densepose, Agnostic, Mask)
            # 注意：它们必须使用完全相同的 batch_index
            video_components = {}
            for key in ['densepose', 'agnostic', 'agnostic_mask']:
                # 假设这些组件在 VIVID 里也是视频文件
                # 如果找不到视频文件，尝试找图片文件夹? 这里先假设是视频
                p = paths[key]
                try:
                    with VideoReader_contextmanager(p, num_threads=2) as vr_comp:
                         # 保护：如果辅助视频长度和原视频不一样怎么办？(VIVID有时会出现这种情况)
                         # 使用取模防止越界
                        comp_len = len(vr_comp)
                        safe_index = [i % comp_len for i in batch_index]
                        frames = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(vr_comp, safe_index))
                        video_components[key] = frames
                except Exception as e:
                    raise ValueError(f"Failed to read {key} video: {p}. Error: {e}")

            # 2.3 组装 Sample
            sample = {}
            sample['input'] = self._process_video_frames(input_frames)
            sample['densepose'] = self._process_video_frames(video_components['densepose'])
            sample['agnostic'] = self._process_video_frames(video_components['agnostic'])
            sample['agnostic_mask'] = self._process_video_frames(video_components['agnostic_mask'], is_mask=True)
            
            # Cloth 始终是图片
            sample['cloth'] = self._process_image(paths['cloth']) # [1, C, H, W]

            text = data_info.get('caption', '')
            if random.random() < self.text_drop_ratio: text = ''
            
            return sample, text, 'video', paths['input']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 增加了重试机制
        attempts = 0
        while attempts < 10:
            try:
                sample_dict, text, dtype, fpath = self.get_batch(idx)
                
                # 最终输出字典解包
                ret = {
                    "pixel_values": sample_dict['input'],         # Input
                    "cloth_pixel_values": sample_dict['cloth'],   # Cloth
                    "densepose_pixel_values": sample_dict['densepose'],
                    "agnostic_pixel_values": sample_dict['agnostic'],
                    "mask_pixel_values": sample_dict['agnostic_mask'],
                    "text": text,
                    "data_type": dtype,
                    "file_name": os.path.basename(fpath)
                }
                return ret
                
            except Exception as e:
                print(f"Error loading idx {idx}: {e}. Retrying...")
                idx = random.randint(0, self.length - 1)
                attempts += 1
        
        raise ValueError("Failed to load batch after 10 attempts")
    
