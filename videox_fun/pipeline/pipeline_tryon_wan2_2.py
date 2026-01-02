import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from diffusers.video_processor import VideoProcessor

# 假设你的模型定义在这些路径，请根据实际情况调整 import
from videox_fun.models import (
    AutoencoderKLWan3_8, 
    WanT5EncoderModel, 
)
from videox_fun.models.wan_transformer3d_tryon import WanTransformer3DTryonModel
from transformers import (
    T5Tokenizer, 
    SiglipVisionModel, 
    SiglipImageProcessor, 
    AutoModel, 
    AutoImageProcessor
)

logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class WanTryOnPipelineOutput(BaseOutput):
    videos: torch.Tensor

class WanTryOnPipeline(DiffusionPipeline):
    r"""
    Pipeline for Virtual Try-On video generation using Wan.
    """

    model_cpu_offload_seq = "text_encoder->siglip_image_encoder->dino_image_encoder->transformer->vae"

    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: WanT5EncoderModel,
        vae: AutoencoderKLWan3_8,
        transformer: WanTransformer3DTryonModel,
        siglip_image_encoder: SiglipVisionModel,
        siglip_image_processor: SiglipImageProcessor,
        dino_image_encoder: AutoModel,
        dino_image_processor: AutoImageProcessor,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            siglip_image_encoder=siglip_image_encoder,
            siglip_image_processor=siglip_image_processor,
            dino_image_encoder=dino_image_encoder,
            dino_image_processor=dino_image_processor,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 16
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor)

    # --------------------------------------------------------------------------
    #  Copied & Adapted from your Training Script (Image Feature Extraction)
    # --------------------------------------------------------------------------
    def _encode_siglip_image_emb(self, siglip_image, device, dtype):
        siglip_image = siglip_image.to(device, dtype=dtype)
        res = self.siglip_image_encoder(siglip_image, output_hidden_states=True)
        siglip_image_embeds = res.last_hidden_state
        # SigLIP shallow features from specific layers
        siglip_image_shallow_embeds = torch.cat([res.hidden_states[i] for i in [7, 13, 26]], dim=1)
        return siglip_image_embeds, siglip_image_shallow_embeds

    def _encode_dinov2_image_emb(self, dinov2_image, device, dtype):
        dinov2_image = dinov2_image.to(device, dtype=dtype)
        res = self.dino_image_encoder(dinov2_image, output_hidden_states=True)
        dinov2_image_embeds = res.last_hidden_state[:, 1:] # Skip CLS token
        # DINO shallow features
        dinov2_image_shallow_embeds = torch.cat([res.hidden_states[i][:, 1:] for i in [9, 19, 29]], dim=1)
        return dinov2_image_embeds, dinov2_image_shallow_embeds

    def get_image_embeds(self, cloth_image_tensor, device, dtype):
        """
        Processing logic copied from vton_collate_fn / encode_image_emb
        Args:
            cloth_image_tensor: (B, C, H, W) range [-1, 1] (Normalized)
        """
        # 1. Un-normalize to [0, 1] for processors
        image_01 = cloth_image_tensor * 0.5 + 0.5

        def process_with_processor(images, processor):
            # images: (B, C, H, W) range [0, 1]
            if isinstance(images, torch.Tensor) and images.dtype == torch.bfloat16: 
                images = images.to(torch.float32)
            # Diffusers processors expect numpy or PIL usually, but here we manually handle tensors
            # Assuming the processors can handle tensors or we use internal normalize
            # To be safe and match training script logic strictly:
            return processor(
                images=images,
                return_tensors="pt",
                do_resize=False,
                do_rescale=False,
                do_normalize=True
            ).pixel_values.to(device, dtype)

        # Step B: Low Res (384x384)
        image_low_res = F.interpolate(image_01, size=(384, 384), mode='bicubic', align_corners=False, antialias=True)

        # Step C: High Res (768x768) + Crops
        image_high_res_full = F.interpolate(image_01, size=(768, 768), mode='bicubic', align_corners=False, antialias=True)
        crops = [
            image_high_res_full[:, :, 0:384, 0:384],
            image_high_res_full[:, :, 0:384, 384:768],
            image_high_res_full[:, :, 384:768, 0:384],
            image_high_res_full[:, :, 384:768, 384:768],
        ]
        image_high_res_crops = torch.stack(crops, dim=1) # (B, 4, C, 384, 384)
        nb_split_image = 4
        image_high_res_flat = rearrange(image_high_res_crops, 'b n c h w -> (b n) c h w')

        # Step D: Low Res Features
        siglip_low_input = process_with_processor(image_low_res, self.siglip_image_processor)
        dino_low_input = process_with_processor(image_low_res, self.dino_image_processor)

        siglip_embeds_low, siglip_shallow_low = self._encode_siglip_image_emb(siglip_low_input, device, dtype)
        dinov2_embeds_low, dinov2_shallow_low = self._encode_dinov2_image_emb(dino_low_input, device, dtype)

        image_embeds_low_res_deep = torch.cat([siglip_embeds_low, dinov2_embeds_low], dim=2)
        image_embeds_low_res_shallow = torch.cat([siglip_shallow_low, dinov2_shallow_low], dim=2)

        # Step E: High Res Features
        siglip_high_input = process_with_processor(image_high_res_flat, self.siglip_image_processor)
        dino_high_input = process_with_processor(image_high_res_flat, self.dino_image_processor)

        siglip_embeds_high, _ = self._encode_siglip_image_emb(siglip_high_input, device, dtype)
        dinov2_embeds_high, _ = self._encode_dinov2_image_emb(dino_high_input, device, dtype)

        # Reshape back
        siglip_high_res_deep = rearrange(siglip_embeds_high, '(b n) l c -> b (n l) c', n=nb_split_image)
        dinov2_high_res_deep = rearrange(dinov2_embeds_high, '(b n) l c -> b (n l) c', n=nb_split_image)
        image_embeds_high_res_deep = torch.cat([siglip_high_res_deep, dinov2_high_res_deep], dim=2)

        return dict(
            image_embeds_low_res_shallow=image_embeds_low_res_shallow,
            image_embeds_low_res_deep=image_embeds_low_res_deep,
            image_embeds_high_res_deep=image_embeds_high_res_deep,
        )

    # --------------------------------------------------------------------------
    #  Mask Preparation (Copied from Training Script)
    # --------------------------------------------------------------------------
    def prepare_vton_mask(self, mask, latents):
        # mask: (B, C, F, H, W) -> assumes C=1 for binary mask
        # latents: (B, C, F_lat, H_lat, W_lat)
        
        # Temporal padding logic from training script (repeat interleave 4)
        # Note: This logic seems specific to how Wan handles temporal tokenization or VAE structure
        mask_padded = torch.cat(
            [
                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                mask[:, :, 1:]
            ], dim=2
        )
        
        b, c, t, h, w = mask_padded.shape
        # Handle case where t is not perfectly divisible by 4 if necessary, 
        # but training script assumes structure.

        mask_view = mask_padded.view(b, c, t // 4, 4, h, w)
        mask_folded = mask_view.permute(0, 3, 1, 2, 4, 5).reshape(b, 4, t // 4, h, w)
        
        target_h, target_w = latents.shape[-2], latents.shape[-1]
        
        # Resize to latent spatial dimension
        mask_final = F.interpolate(
            mask_folded, 
            size=(mask_folded.shape[2], target_h, target_w), 
            mode="nearest"
        )
        return mask_final

    # --------------------------------------------------------------------------
    #  Standard Pipeline Methods
    # --------------------------------------------------------------------------
    def _get_t5_prompt_embeds(self, prompt, max_sequence_length=512, device=None, dtype=None):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        
        prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def encode_prompt(self, prompt, negative_prompt, device=None, dtype=None):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        prompt_embeds = self._get_t5_prompt_embeds(prompt, device=device, dtype=dtype)
        
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        
        negative_prompt_embeds = self._get_t5_prompt_embeds(negative_prompt, device=device, dtype=dtype)
        
        return prompt_embeds, negative_prompt_embeds

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        frames = self.vae.decode(latents.to(self.vae.dtype)).sample
        frames = (frames / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        frames = frames.cpu().float().numpy()
        return frames

    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):
        # Calculate latent shape based on VAE compression
        # Wan3_8 usually has spatial stride 8, temporal stride 4
        temporal_compression = 4
        spatial_compression = 16
        
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // temporal_compression + 1,
            height // spatial_compression,
            width // spatial_compression,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: torch.Tensor,            # Agnostic Image (Person with cloth masked out) [B, C, F, H, W]
        mask_image: torch.Tensor,       # Binary Mask (1 for inpainting area) [B, 1, F, H, W]
        densepose_image: torch.Tensor,  # Densepose [B, C, F, H, W]
        cloth_image: torch.Tensor,      # Cloth Image [B, C, 1, H, W] or [B, C, H, W]
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 5.0,
        height: int = 480,
        width: int = 720, # Should match training aspect ratio logic
        num_frames: int = 49,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: str = "numpy",
        return_dict: bool = True,
    ):
        
        # 0. Setup
        device = self._execution_device
        dtype = self.text_encoder.dtype
        
        # Ensure inputs are tensors and normalized to [-1, 1] if they aren't already
        # Here we assume inputs are already preprocessed tensors in range [-1, 1] or [0, 1] based on training script expectations.
        # The training script collate_fn implies inputs are tensor [C, H, W]
        
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # 1. Encode Prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt, device, dtype)
        
        # CFG processing
        if guidance_scale > 1.0:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            prompt_embeds = prompt_embeds

        # 2. Encode VAE Inputs (Agnostic, Densepose, Cloth)
        # Helper to encode video to latents
        def encode_vae(video_tensor):
            # video_tensor: (B, C, F, H, W)
            # VAE expects: (B, C, F, H, W)
            video_tensor = video_tensor.to(device=device, dtype=self.vae.dtype)
            # Using slice logic from training script if necessary, or direct encode
            # Training script: vae.encode(video_tensor)[0].sample()
            dist = self.vae.encode(video_tensor)[0]
            return dist.sample()

        # Prepare Agnostic Latents (Masked Video)
        mask_latents = encode_vae(image)
        
        # Prepare Densepose Latents
        densepose_latents = encode_vae(densepose_image)
        
        # Prepare Cloth Latents (VAE Encoded)
        # Ensure cloth has time dimension for VAE
        if cloth_image.ndim == 4:
            cloth_image_video = cloth_image.unsqueeze(2) # B, C, 1, H, W
        else:
            cloth_image_video = cloth_image
        cloth_latents = encode_vae(cloth_image_video)
        
        # 3. Encode Cloth Visual Features (SigLIP + DINO)
        # Input to get_image_embeds should be (B, C, H, W)
        if cloth_image.ndim == 5:
            cloth_frame0 = cloth_image[:, :, 0, :, :]
        else:
            cloth_frame0 = cloth_image
        
        subject_image_embeds_dict = self.get_image_embeds(cloth_frame0, device, dtype)
        
        # Duplicate image embeds for CFG
        if guidance_scale > 1.0:
            for k, v in subject_image_embeds_dict.items():
                subject_image_embeds_dict[k] = torch.cat([v] * 2, dim=0)

        # 4. Prepare Generation Latents (Noise)
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            num_frames,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare Inpainting Mask condition (Binary Mask resizing)
        # mask_image should be (B, 1, F, H, W)
        mask_condition = self.prepare_vton_mask(mask_image.to(device, dtype), latents)
        print("Mask condition shape:", mask_condition.shape)
        print("mask latents shape:", mask_latents.shape)
        print("densepose latents shape:", densepose_latents.shape)
        # 6. Concatenate Conditionings for 'y' input
        # Training script: inpaint_latents = torch.cat([mask_values, mask_latents, densepose_latents], dim=1)
        inpaint_latents = torch.cat([mask_condition, mask_latents, densepose_latents], dim=1)
        
        # Duplicate for CFG
        if guidance_scale > 1.0:
            inpaint_latents = torch.cat([inpaint_latents] * 2, dim=0)
            cloth_latents = torch.cat([cloth_latents] * 2, dim=0)

        # 7. Prepare Scheduler
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, 
            num_inference_steps, 
            device, 
            timesteps, 
            mu=1.0  # Wan 的 FlowMatch 默认 mu=1.0
        )

        # 8. Denoising Loop
        # Calculate seq_len for Transformer (3D RoPE requirement)
        target_shape = latents.shape
        # Patch size usually (1, 2, 2) for Wan
        patch_size = self.transformer.config.patch_size
        seq_len = math.ceil((target_shape[3] * target_shape[4]) / (patch_size[1] * patch_size[2]) * target_shape[2])

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                if hasattr(self.scheduler, "scale_model_input"):
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                
                # Broadcast time
                timestep = t.expand(latent_model_input.shape[0])

                # Forward Pass
                # Matching Training Script: 
                # noise_pred = transformer3d(x, seq_len, context, t, y, cloth_latents, subject_image_embeds_dict)
                noise_pred = self.transformer(
                    x=latent_model_input,
                    seq_len=seq_len,
                    context=prompt_embeds,
                    t=timestep,
                    y=inpaint_latents,
                    cloth_latents=cloth_latents,
                    subject_image_embeds_dict=subject_image_embeds_dict
                )

                # CFG
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                # (Optional) Repainting Strategy: Paste back original pixels in masked area?
                # Usually in learned TryOn, the model handles it via 'y'. 
                # But to enforce background consistency:
                # latents = (1 - mask_condition) * mask_latents + mask_condition * latents
                # Note: mask_latents needs to be noise-injected if doing strict inpainting loop, 
                # but Wan architecture here seems to treat it as generation conditioned on y.
                # Let's trust the model output for now as per training script logic.

                progress_bar.update()

        # 9. Decode
        if output_type == "numpy":
            video = self.decode_latents(latents)
        elif not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            video = torch.from_numpy(video)

        return WanTryOnPipelineOutput(videos=video)