import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from peft import PeftModel
from PIL import Image
import torch.nn.functional as F  # 导入 F 用于插值
import logging

# -----------------------------------------------------------------------------
# 导入你的模块
# -----------------------------------------------------------------------------
from videox_fun.models import AutoencoderKLWan3_8, WanT5EncoderModel
from videox_fun.models.wan_transformer3d_tryon import WanTransformer3DTryonModel
from videox_fun.pipeline.pipeline_tryon_wan2_2 import WanTryOnPipeline
from videox_fun.data.tryon_video import TryOnDataset
from transformers import AutoTokenizer, SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoImageProcessor
from diffusers import FlowMatchEulerDiscreteScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    return {k: v for k, v in kwargs.items() if k in valid_params}

# =========================================================================
# 修复后的图片保存函数
# =========================================================================
def save_generated_image(video_frames, save_path):
    """
    保存单帧图像
    video_frames: numpy array [B, C, F, H, W] 或 [C, F, H, W]
    """
    if video_frames.ndim == 5:
        video_frames = video_frames[0]
    if video_frames.ndim == 4:
        video_frames = video_frames[:, 0, :, :] 
    if video_frames.shape[0] in [1, 3]: 
        video_frames = np.transpose(video_frames, (1, 2, 0)) 
    if video_frames.shape[-1] == 1:
        video_frames = video_frames.squeeze(-1)

    frame = (video_frames * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(frame).save(save_path)

def save_input_tensor(tensor, save_path):
    """
    保存输入 Tensor 用于对比
    """
    if tensor.ndim == 5: img = tensor[0, :, 0, :, :] 
    elif tensor.ndim == 4: img = tensor[:, 0, :, :] 
    elif tensor.ndim == 3: img = tensor
    else: return

    # 先 float() 解决 bfloat16 问题，再 cpu()
    img = img.float().cpu().permute(1, 2, 0).numpy() # [H, W, C]
    
    # 反归一化 [-1, 1] -> [0, 1]
    if img.min() < 0:
        img = (img / 2 + 0.5)
        
    img = (img * 255).clip(0, 255).astype(np.uint8)
    
    if img.shape[2] == 1: img = img.squeeze(2) # Mask 处理
    Image.fromarray(img).save(save_path)

def main():
    parser = argparse.ArgumentParser()
    # --- 参数命名与训练脚本对齐 ---
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Wan2.1 Base Model Root")
    parser.add_argument("--config_path", type=str, default="config/train_mymodel/train_wan_2-2.yaml")
    parser.add_argument("--image_encoder_path", type=str, required=True, help="SigLIP path")
    parser.add_argument("--image_encoder_2_path", type=str, required=True, help="DINOv2 path")
    
    # 推理特有参数
    parser.add_argument("--lora_checkpoint_path", type=str, required=True)
    parser.add_argument("--test_data_dir", type=str, required=True)
    parser.add_argument("--test_data_meta", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="inference_results_strict_load")
    
    # 【修改点】默认分辨率改为 512(H) x 384(W) 竖屏
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=384)
    
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=1000)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16
    
    # 1. Load Config
    config = OmegaConf.load(args.config_path)
    logger.info(f"🚀 Loading models from {args.pretrained_model_name_or_path}...")

    # Load Encoders & VAE
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')))
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        torch_dtype=weight_dtype
    ).eval()

    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).eval() 

    # Load Transformer
    transformer = WanTransformer3DTryonModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=False    
    )

    # Load Image Encoders
    siglip_image_encoder = SiglipVisionModel.from_pretrained(args.image_encoder_path, dtype=weight_dtype).eval()
    siglip_image_processor = SiglipImageProcessor.from_pretrained(args.image_encoder_path)
    dino_image_encoder = AutoModel.from_pretrained(args.image_encoder_2_path, dtype=weight_dtype).eval()
    dino_image_processor = AutoImageProcessor.from_pretrained(args.image_encoder_2_path)
    dino_image_processor.crop_size = dict(height=384, width=384)
    dino_image_processor.size = dict(shortest_edge=384)

    # 3. Load LoRA & Merge
    print(f"♻️ Loading LoRA from {args.lora_checkpoint_path}...")
    transformer = PeftModel.from_pretrained(transformer, args.lora_checkpoint_path)
    # 【关键】Merge 确保权重生效
    transformer = transformer.merge_and_unload()
    transformer.eval()

    # 5. Pipeline
    scheduler = FlowMatchEulerDiscreteScheduler(**filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs'])))

    pipeline = WanTryOnPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer,
        siglip_image_encoder=siglip_image_encoder, siglip_image_processor=siglip_image_processor,
        dino_image_encoder=dino_image_encoder, dino_image_processor=dino_image_processor,
        scheduler=scheduler
    )
    pipeline.to(device, dtype=weight_dtype)
    
    # 6. Dataset
    dataset = TryOnDataset(
        ann_path=args.test_data_meta,
        data_root=args.test_data_dir,
        video_sample_stride=1,
        video_sample_n_frames=1, 
        video_repeat=1, 
        filter_type="image"
    )
    print(f"📂 Found {len(dataset)} samples. Target Res: {args.height}x{args.width}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # 7. Inference Loop
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        base_name = i
        
        # =========================================================
        # 【修改点】强制 Resize 函数
        # =========================================================
# =========================================================
        # 【修改点】支持 5D 输入的强制 Resize 函数
        # =========================================================
        def prepare_tensor(t, is_mask=False):
            t = t.to(device, weight_dtype)
            
            # 1. 维度标准化: 统一处理成 5D [B, C, F, H, W]
            # Dataset 如果返回 [C, H, W] -> 变成 [1, 1, C, H, W] -> permute
            if t.ndim == 3: # [C, H, W]
                t = t.unsqueeze(0) # [1, C, H, W] (视为 T=1)
                
            # Dataset 如果返回 [T, C, H, W] -> [1, T, C, H, W]
            if t.ndim == 4:
                t = t.unsqueeze(0)

            # 此时 t 是 [1, T, C, H, W]，我们需要 [1, C, T, H, W] (Pipeline 格式)
            t = t.permute(0, 2, 1, 3, 4) 

            # 2. 强制 Resize
            # 注意：t 现在是 5D [B, C, F, H, W]
            if t.shape[-2] != args.height or t.shape[-1] != args.width:
                mode = 'nearest' if is_mask else 'bilinear'
                
                # 【核心修复】将 5D 压扁为 4D 进行插值
                b, c, f, h, w = t.shape
                
                # [B, C, F, H, W] -> [B, F, C, H, W] -> [B*F, C, H, W]
                t_flattened = t.transpose(1, 2).reshape(b * f, c, h, w)
                
                # 对 spatial 维度 (H, W) 进行 resize
                t_resized = F.interpolate(
                    t_flattened, 
                    size=(args.height, args.width), 
                    mode=mode
                )
                
                # 变回 5D: [B*F, C, H', W'] -> [B, F, C, H', W'] -> [B, C, F, H', W']
                t = t_resized.reshape(b, f, c, args.height, args.width).transpose(1, 2)
                
            return t

        # 准备所有输入 (全部 Resize 到 512x384)
        agnostic = prepare_tensor(sample["agnostic_pixel_values"])
        mask = prepare_tensor(sample["mask_pixel_values"], is_mask=True)
        densepose = prepare_tensor(sample["densepose_pixel_values"])
        pixel = prepare_tensor(sample["pixel_values"]) # GT
        
        # Cloth 特殊处理
        cloth_raw = sample["cloth_pixel_values"]
        # Cloth 也要过一遍 prepare_tensor 确保尺寸一致 (虽然它主要进 ImageEncoder，但 Pipeline 可能用到)
        # 即使 cloth 是 [C, H, W]，prepare_tensor 也能处理
        cloth = prepare_tensor(cloth_raw) 
        
        prompt = sample["text"]
        
        # 保存输入 (Resize 后的)
        save_input_tensor(agnostic, os.path.join(args.output_dir, f"{base_name}_1_agnostic.png"))
        save_input_tensor(cloth, os.path.join(args.output_dir, f"{base_name}_2_cloth.png"))
        save_input_tensor(densepose, os.path.join(args.output_dir, f"{base_name}_3_densepose.png"))
        # save_input_tensor(pixel, os.path.join(args.output_dir, f"{base_name}_4_gt.png")) # 可选

        # Generate
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                image=agnostic,
                mask_image=mask,
                densepose_image=densepose,
                cloth_image=cloth,
                num_frames=1,
                # 显式传入目标宽高
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="numpy"
            )

        save_generated_image(output.videos, os.path.join(args.output_dir, f"{base_name}_5_result.png"))
        
        if i == 10:
            break

    print("✅ Inference finished!")

if __name__ == "__main__":
    main()