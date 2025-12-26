import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging
from diffusers.video_processor import VideoProcessor
from diffusers.image_processor import VaeImageProcessor 
from einops import rearrange
from PIL import Image
from transformers import AutoImageProcessor, SiglipImageProcessor
from diffusers.utils.torch_utils import randn_tensor


from ..models import (AutoencoderKLWan, AutoTokenizer, CLIPModel,
                      WanT5EncoderModel)
from ..models.wan_transformer3d_tryon import WanTransformer3DTryonModel 

logger = logging.get_logger(__name__)

@dataclass
class WanVTONPipelineOutput(BaseOutput):
    videos: torch.Tensor

class WanVTONPipeline(DiffusionPipeline):
    _optional_components = []
    model_cpu_offload_seq = "text_encoder->clip_image_encoder->siglip_image_encoder->dino_image_encoder->transformer->vae"

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan,
        transformer: WanTransformer3DTryonModel,
        clip_image_encoder: CLIPModel,
        siglip_image_encoder: Any,
        dino_image_encoder: Any,
        scheduler: FlowMatchEulerDiscreteScheduler,
        siglip_image_processor: Optional[SiglipImageProcessor] = None,
        dino_image_processor: Optional[AutoImageProcessor] = None,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            vae=vae, 
            transformer=transformer, 
            clip_image_encoder=clip_image_encoder,
            siglip_image_encoder=siglip_image_encoder,
            dino_image_encoder=dino_image_encoder,
            scheduler=scheduler
        )
        self.siglip_image_processor = siglip_image_processor
        self.dino_image_processor = dino_image_processor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae.spatial_compression_ratio)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae.spatial_compression_ratio, 
            do_normalize=False, 
            do_binarize=True, 
            do_convert_grayscale=True
        )

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def _process_feature_extractor_input(self, image, processor):
        if isinstance(image, Image.Image): image = TF.to_tensor(image)
        if isinstance(image, torch.Tensor) and image.ndim == 3: image = image.unsqueeze(0)
        if image.dtype == torch.bfloat16: image = image.to(torch.float32)
        return processor(
            images=image, return_tensors="pt", do_resize=False, do_center_crop=False, 
            do_rescale=False, do_normalize=True
        ).pixel_values.to(self.device, self.weight_dtype)

    def _get_visual_embeds(self, cloth_image_tensor):
        image_01 = cloth_image_tensor * 0.5 + 0.5
        image_low_res = F.interpolate(image_01, size=(384, 384), mode='bicubic', align_corners=False, antialias=True)
        image_high_res_full = F.interpolate(image_01, size=(768, 768), mode='bicubic', align_corners=False, antialias=True)
        crops = [
            image_high_res_full[:, :, 0:384, 0:384],
            image_high_res_full[:, :, 0:384, 384:768],
            image_high_res_full[:, :, 384:768, 0:384],
            image_high_res_full[:, :, 384:768, 384:768],
        ]
        image_high_res_crops = torch.stack(crops, dim=1)
        nb_split_image = 4
        image_high_res_flat = rearrange(image_high_res_crops, 'b n c h w -> (b n) c h w')

        siglip_low = self._process_feature_extractor_input(image_low_res, self.siglip_image_processor)
        dino_low = self._process_feature_extractor_input(image_low_res, self.dino_image_processor)
        siglip_high = self._process_feature_extractor_input(image_high_res_flat, self.siglip_image_processor)
        dino_high = self._process_feature_extractor_input(image_high_res_flat, self.dino_image_processor)

        with torch.no_grad():
            res_s_low = self.siglip_image_encoder(siglip_low, output_hidden_states=True)
            res_d_low = self.dino_image_encoder(dino_low, output_hidden_states=True)
            res_s_high = self.siglip_image_encoder(siglip_high, output_hidden_states=True)
            res_d_high = self.dino_image_encoder(dino_high, output_hidden_states=True)

        img_emb_low_deep = torch.cat([res_s_low.last_hidden_state, res_d_low.last_hidden_state[:, 1:]], dim=2)
        siglip_shallow = torch.cat([res_s_low.hidden_states[i] for i in [7, 13, 26]], dim=1)
        dino_shallow = torch.cat([res_d_low.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        img_emb_low_shallow = torch.cat([siglip_shallow, dino_shallow], dim=2)

        siglip_high_deep = rearrange(res_s_high.last_hidden_state, '(b n) l c -> b (n l) c', n=nb_split_image)
        dino_high_deep = rearrange(res_d_high.last_hidden_state[:, 1:], '(b n) l c -> b (n l) c', n=nb_split_image)
        img_emb_high_deep = torch.cat([siglip_high_deep, dino_high_deep], dim=2)

        return dict(
            image_embeds_low_res_shallow=img_emb_low_shallow,
            image_embeds_low_res_deep=img_emb_low_deep,
            image_embeds_high_res_deep=img_emb_high_deep,
        )

    def prepare_vton_latents(self, agnostic_video_tensor, densepose_video_tensor, mask_video_tensor, num_frames, height, width):
        """
        输入已经是预处理好的 Tensor:
        agnostic/densepose: [B, C, F, H, W] in [-1, 1]
        mask: [B, 1, F, H, W] in [0, 1]
        """
        device = self.device
        dtype = self.vae.dtype

        # 1. VAE Encode Agnostic (Batch Encode)
        # 输入: [1, 3, 49, H, W]
        dist = self.vae.encode(agnostic_video_tensor.to(device, dtype))[0]
        agnostic_latents = dist.mode() # [1, 16, T_lat, H_lat, W_lat]

        # 2. VAE Encode Densepose
        dist = self.vae.encode(densepose_video_tensor.to(device, dtype))[0]
        densepose_latents = dist.mode()

        # 3. 处理 Mask
        # Wan VAE Temporal compression = 4
        target_h, target_w = height // 8, width // 8
        latent_frames = agnostic_latents.shape[2] 

        mask_tensor = mask_video_tensor.to(device, dtype)
        
        # 空间 Resize
        mask_resized = F.interpolate(mask_tensor, size=(num_frames, target_h, target_w), mode='nearest')
        
        # 时间 Resize
        mask_final = F.interpolate(mask_resized, size=(latent_frames, target_h, target_w), mode='nearest')
        
        # Channel Repeat (1 -> 4)
        mask_final = mask_final.repeat(1, 4, 1, 1, 1)

        # 4. Concatenate
        inpaint_latents = torch.cat([mask_final, agnostic_latents, densepose_latents], dim=1)
        
        return inpaint_latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        cloth_image: Image.Image,
        agnostic_video: List[Image.Image],
        densepose_video: List[Image.Image],
        mask_video: List[Image.Image],
        num_frames: int = 49,
        height: int = 768,
        width: int = 576,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: int = 1,
        negative_prompt: Optional[str] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        self.weight_dtype = self.transformer.dtype
        device = self._execution_device

        # === 1. 预处理输入数据 ===
        
        # A. Cloth (静态图) -> [1, 3, H, W]
        # ImageProcessor 输出 (1, 3, H, W)
        cloth_tensor = self.image_processor.preprocess(cloth_image, height=height, width=width).to(device, self.weight_dtype)
        
        # B. Agnostic & Densepose (视频) -> [1, 3, F, H, W]
        # [修正点]: VideoProcessor 已经把 Channel 放在 Frame 前面了，直接用
        # 注意要调用 preprocess_video 而不是 preprocess
        agnostic_tensor = self.video_processor.preprocess_video(agnostic_video, height=height, width=width).to(device, self.weight_dtype)
        densepose_tensor = self.video_processor.preprocess_video(densepose_video, height=height, width=width).to(device, self.weight_dtype)

        # C. Mask (视频) -> [1, 1, F, H, W]
        # MaskProcessor 是 VaeImageProcessor，它处理 List[Image] 会返回 [F, 1, H, W]
        mask_tensor = self.mask_processor.preprocess(mask_video, height=height, width=width).to(device, self.weight_dtype)
        # 我们需要把它变成视频格式 [1, 1, F, H, W]
        mask_tensor = mask_tensor.permute(1, 0, 2, 3).unsqueeze(0) # [F, 1, H, W] -> [1, F, H, W] -> [1, 1, F, H, W]
        
        # ========================

        # 2. 提取特征 (SigLIP + DINO)
        subject_embeds = self._get_visual_embeds(cloth_tensor)
        clip_context = self.clip_image_encoder(cloth_tensor)

        # 3. Text Encoding
        prompt_embeds, neg_embeds = self.encode_prompt(
            prompt, negative_prompt, do_classifier_free_guidance=(guidance_scale > 1.0),
            num_videos_per_prompt=num_videos_per_prompt, 
            device=device, dtype=self.weight_dtype
        )
        if guidance_scale > 1.0:
            context = torch.cat([neg_embeds, prompt_embeds])
            clip_context = torch.cat([clip_context] * 2)
            for k in subject_embeds: subject_embeds[k] = torch.cat([subject_embeds[k]] * 2)
        else:
            context = prompt_embeds

        y_latents = self.prepare_vton_latents(
            agnostic_tensor, densepose_tensor, mask_tensor, 
            num_frames, height, width
        )
        if guidance_scale > 1.0:
            y_latents = torch.cat([y_latents] * 2)

        total_batch_size = batch_size * num_videos_per_prompt
        y_latents = y_latents.repeat(total_batch_size, 1, 1, 1, 1)
        clip_context = clip_context.repeat(total_batch_size, 1, 1)
        for k in subject_embeds:
            if subject_embeds[k].shape[0] == 1:
                subject_embeds[k] = subject_embeds[k].repeat(total_batch_size, 1, 1)

        if guidance_scale > 1.0:
            context = torch.cat([neg_embeds, prompt_embeds])
            clip_context = torch.cat([clip_context] * 2)
            y_latents = torch.cat([y_latents] * 2)
            for k in subject_embeds: 
                subject_embeds[k] = torch.cat([subject_embeds[k]] * 2)
        else:
            context = prompt_embeds

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        latent_channels = self.vae.config.latent_channels
        # 计算 Latent 空间下的帧数和宽高
        # Wan VAE: Temporal / 4, Spatial / 8
        lat_h = height // self.vae.spatial_compression_ratio
        lat_w = width // self.vae.spatial_compression_ratio
        lat_f = (num_frames - 1) // self.vae.temporal_compression_ratio + 1
        
        shape = (total_batch_size, latent_channels, lat_f, lat_h, lat_w)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=self.weight_dtype)
        # 6. 采样循环
        # 计算 Transformer 需要的 seq_len (用于 Attention Mask)
        # Patch Size 通常是 (1, 2, 2)
        p_t, p_h, p_w = self.transformer.config.patch_size
        seq_len = math.ceil((lat_h * lat_w) / (p_h * p_w) * lat_f)

        for t in self.progress_bar(timesteps):
            latent_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # 这里的 latent_input 已经是 [B, C, F, H, W]，无需 rearrange
            noise_pred = self.transformer(
                x=latent_input,
                t=t.expand(latent_input.shape[0]),
                context=context,
                seq_len=seq_len,
                y=y_latents,
                clip_fea=clip_context,
                subject_image_embeds_dict=subject_embeds
            )

            if guidance_scale > 1.0:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 7. 解码
        video = self.decode_latents(latents)
        return WanVTONPipelineOutput(videos=torch.from_numpy(video))