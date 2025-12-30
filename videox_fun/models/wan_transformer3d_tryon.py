# Modified from https://github.com/ali-vilab/VACE/blob/main/vace/models/wan/wan_vace.py
# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import os
import math
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.utils import is_torch_version

from .wan_transformer3d import (WanAttentionBlock, WanTransformer3DModel, Wan2_2Transformer3DModel, WanRMSNorm , WanSelfAttention,
                                sinusoidal_embedding_1d)
from ..utils import cfg_skip
from .attention_utils import attention

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class WanTransformer3DTryonModel(WanTransformer3DModel):
    def __init__(self,                  
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__(model_type,patch_size,text_len,in_dim,dim,ffn_dim,freq_dim,text_dim,out_dim,num_heads,num_layers,window_size,qk_norm,cross_attn_norm,eps)

        self.subject_image_proj_model = CrossLayerCrossScaleProjector(output_dim=self.dim, timestep_in_dim=self.freq_dim)

        self.cloth_dim = 48
        self.cloth_patch_embedding = zero_module(nn.Conv3d(self.cloth_dim, dim, kernel_size=patch_size, stride=patch_size))
        with torch.no_grad():
            self.cloth_patch_embedding.weight.copy_(self.patch_embedding.weight.data[:, :self.cloth_dim, :, :, :])
            self.cloth_patch_embedding.bias.copy_(self.patch_embedding.bias)

        cross_attn_type = 'i2v_cross_attn' 
        self.blocks = nn.ModuleList([
            WanAttentionBlockTryon(
                                    cross_attn_type, 
                                    self.dim, 
                                    self.ffn_dim,
                                    self.num_heads,
                                    self.window_size,
                                    self.qk_norm,
                                    self.cross_attn_norm,
                                    self.eps
                                )
            for _ in range(self.num_layers)
        ])


    @cfg_skip()
    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        subject_image_embeds_dict=None,
        y_camera=None,
        full_ref=None,
        subject_ref=None,
        cond_flag=True,
        cloth_latents=None,
    ):
        t_adapter = t.to(dtype=x.dtype)
        t_adapter = t_adapter*1000.0
        ip_hidden_states = self.subject_image_proj_model(
            subject_image_embeds_dict['image_embeds_low_res_shallow'],
            subject_image_embeds_dict['image_embeds_low_res_deep'],
            subject_image_embeds_dict['image_embeds_high_res_deep'],
            timesteps=t_adapter, 
            need_temb=True
        )[0]

        device = self.patch_embedding.weight.device
        dtype = x.dtype
        if self.freqs.device != device and torch.device(type="meta") != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        if cloth_latents is not None:
            # cloth embeddings
            cloth_latents = self.cloth_patch_embedding(cloth_latents)
            cloth_latents = cloth_latents.flatten(2).transpose(1, 2) # B L C 

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # add control adapter
        if self.control_adapter is not None and y_camera is not None:
            y_camera = self.control_adapter(y_camera)
            x = [u + v for u, v in zip(x, y_camera)]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        if self.ref_conv is not None and full_ref is not None:
            full_ref = self.ref_conv(full_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += full_ref.size(1)
            x = [torch.concat([_full_ref.unsqueeze(0), u], dim=1) for _full_ref, u in zip(full_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([padding, t], dim=1)

        if subject_ref is not None:
            subject_ref_frames = subject_ref.size(2)
            subject_ref = self.patch_embedding(subject_ref).flatten(2).transpose(1, 2)
            grid_sizes = torch.stack([torch.tensor([u[0] + subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
            seq_len += subject_ref.size(1)
            x = [torch.concat([u, _subject_ref.unsqueeze(0)], dim=1) for _subject_ref, u in zip(subject_ref, x)]
            if t.dim() != 1 and t.size(1) < seq_len:
                pad_size = seq_len - t.size(1)
                last_elements = t[:, -1].unsqueeze(1)
                padding = last_elements.repeat(1, pad_size)
                t = torch.cat([t, padding], dim=1)
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        if self.sp_world_size > 1:
            seq_len = int(math.ceil(seq_len / self.sp_world_size)) * self.sp_world_size
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

 # time embeddings
        with torch.amp.autocast(device_type="cuda",dtype=torch.float32):
            if t.dim() != 1:
                if t.size(1) < seq_len:
                    pad_size = seq_len - t.size(1)
                    last_elements = t[:, -1].unsqueeze(1)
                    padding = last_elements.repeat(1, pad_size)
                    t = torch.cat([t, padding], dim=1)
                bt = t.size(0)
                ft = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            ft).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            else:
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))

            # assert e.dtype == torch.float32 and e0.dtype == torch.float32
            e0 = e0.to(dtype)
            e = e.to(dtype)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # Context Parallel
        if self.sp_world_size > 1:
            x = torch.chunk(x, self.sp_world_size, dim=1)[self.sp_world_rank]
            if t.dim() != 1:
                e0 = torch.chunk(e0, self.sp_world_size, dim=1)[self.sp_world_rank]
                e = torch.chunk(e, self.sp_world_size, dim=1)[self.sp_world_rank]
        
        # TeaCache
        if self.teacache is not None:
            if cond_flag:
                if t.dim() != 1:
                    modulated_inp = e0[:, -1, :]
                else:
                    modulated_inp = e0
                skip_flag = self.teacache.cnt < self.teacache.num_skip_start_steps
                if skip_flag:
                    self.should_calc = True
                    self.teacache.accumulated_rel_l1_distance = 0
                else:
                    if cond_flag:
                        rel_l1_distance = self.teacache.compute_rel_l1_distance(self.teacache.previous_modulated_input, modulated_inp)
                        self.teacache.accumulated_rel_l1_distance += self.teacache.rescale_func(rel_l1_distance)
                    if self.teacache.accumulated_rel_l1_distance < self.teacache.rel_l1_thresh:
                        self.should_calc = False
                    else:
                        self.should_calc = True
                        self.teacache.accumulated_rel_l1_distance = 0
                self.teacache.previous_modulated_input = modulated_inp
                self.teacache.should_calc = self.should_calc
            else:
                self.should_calc = self.teacache.should_calc
        
        # TeaCache
        if self.teacache is not None:
            if not self.should_calc:
                previous_residual = self.teacache.previous_residual_cond if cond_flag else self.teacache.previous_residual_uncond
                x = x + previous_residual.to(x.device)[-x.size()[0]:,]
            else:
                ori_x = x.clone().cpu() if self.teacache.offload else x.clone()

                for block in self.blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs)

                            return custom_forward
                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x,
                            e0,
                            seq_lens,
                            grid_sizes,
                            self.freqs,
                            context,
                            context_lens,
                            ip_hidden_states,
                            cloth_latents,
                            t,
                            dtype,
                            **ckpt_kwargs,
                        )
                    else:
                        # arguments
                        kwargs = dict(
                            e=e0,
                            seq_lens=seq_lens,
                            grid_sizes=grid_sizes,
                            freqs=self.freqs,
                            context=context,
                            context_lens=context_lens,
                            dtype=dtype,
                            ip_hidden_states=ip_hidden_states,
                            cloth_latents=cloth_latents,
                            t=t  
                        )
                        x = block(x, **kwargs)
                    
                if cond_flag:
                    self.teacache.previous_residual_cond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
                else:
                    self.teacache.previous_residual_uncond = x.cpu() - ori_x if self.teacache.offload else x - ori_x
        else:
            for block in self.blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward
                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        e0,
                        seq_lens,
                        grid_sizes,
                        self.freqs,
                        context,
                        context_lens,
                        ip_hidden_states,
                        cloth_latents,
                        t,
                        dtype,
                        **ckpt_kwargs,
                    )
                else:
                    # arguments
                    kwargs = dict(
                        e=e0,
                        seq_lens=seq_lens,
                        grid_sizes=grid_sizes,
                        freqs=self.freqs,
                        context=context,
                        context_lens=context_lens,
                        dtype=dtype,
                        ip_hidden_states=ip_hidden_states,
                        cloth_latents=cloth_latents,
                        t=t  
                    )
                    x = block(x, **kwargs)

        # head
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward
            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            x = torch.utils.checkpoint.checkpoint(create_custom_forward(self.head), x, e, **ckpt_kwargs)
        else:
            x = self.head(x, e)

        if self.sp_world_size > 1:
            x = self.all_gather(x, dim=1)

        if self.ref_conv is not None and full_ref is not None:
            full_ref_length = full_ref.size(1)
            x = x[:, full_ref_length:]
            grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        if subject_ref is not None:
            subject_ref_length = subject_ref.size(1)
            x = x[:, :-subject_ref_length]
            grid_sizes = torch.stack([torch.tensor([u[0] - subject_ref_frames, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        x = torch.stack(x)
        if self.teacache is not None and cond_flag:
            self.teacache.cnt += 1
            if self.teacache.cnt == self.teacache.num_steps:
                self.teacache.reset()
        return x
        

class Adapter(nn.Module):
    """Lightweight residual adapter"""
    def __init__(self, dim, adapter_dim=64):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, dim)
        )

    def forward(self, x):
        return self.ffn(x)

class WanAttentionBlockTryon(WanAttentionBlock):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, window_size=..., qk_norm=True, cross_attn_norm=False, eps=0.000001):
        super().__init__(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
        self.cross_attn = WanT2VCrossAttentionTryon(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps
        )
    
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        ip_hidden_states,
        cloth_latents,
        t=0,
        dtype=torch.float32
    ):
        e = (self.modulation + e).chunk(6, dim=1)
        # 这一步将时间嵌入 e 与可学习的调制参数 self.modulation 结合，并将其拆分为 6 个 (shift, scale) 对，用于后续的 LayerNorm。
        # e = (self.modulation + e).chunk(6, dim=1)
        # self.modulation: [1, 6, C]
        # e (输入): [B, 6, C]
        # self.modulation + e: [B, 6, C] (利用广播机制将 self.modulation 加到批次中的每个样本上)
        # .chunk(6, dim=1): 沿 dim=1 将张量分割成 6 块。
        # e (新): 变成一个包含 6 个张量的 tuple (元组)，其中 e[0] 到 e[5] 的 shape 都是 [B, 1, C]。


        # self-attention
        temp_x = self.norm1(x) * (1 + e[1]) + e[0]
        temp_x = temp_x.to(dtype)
        # self.norm1(x): [B, L, C] (LayerNorm 不改变 shape)
        # temp_x: [B, L, C]

        y = self.self_attn(temp_x, seq_lens, grid_sizes, freqs, dtype,t) # un dim size is [B L C]
        x = x + y * e[2]
        # self.self_attn 是 WanSelfAttention。
        # WanSelfAttention 内部计算 q, k, v，应用 RoPE，执行注意力，然后通过输出投影 self.o。
        # 它接收 [B, L, C]，输出的 y shape 仍然是 [B, L, C]

        def cross_attn_ffn(x, context, ip_hidden_states, context_lens, cloth_latents, e):
            x = x + self.cross_attn(self.norm3(x), context, ip_hidden_states, context_lens, cloth_latents, t)
            temp_x = self.norm2(x) * (1 + e[4]) + e[3]
            temp_x = temp_x.to(dtype)
            
            y = self.ffn(temp_x)
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, ip_hidden_states, context_lens,cloth_latents, e)
        return x

class WanT2VCrossAttentionTryon(WanSelfAttention):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        # self.k_img = nn.Linear(dim, dim)
        # self.v_img = nn.Linear(dim, dim)
        # self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.k_ip = nn.Linear(dim, dim)
        self.v_ip = nn.Linear(dim, dim)
        self.norm_k_ip = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.adapter_cloth = Adapter(dim)

    def forward(self, x, context,ip_hidden_states, context_lens, cloth_latents, t=0,dtype=torch.bfloat16):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        # context_img = context[:, :257]
        # context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x.to(dtype))).view(b, -1, n, d)
        k = self.norm_k(self.k(context.to(dtype))).view(b, -1, n, d)
        v = self.v(context.to(dtype)).view(b, -1, n, d)
        # k_img = self.norm_k_img(self.k_img(context_img.to(dtype))).view(b, -1, n, d)
        # v_img = self.v_img(context_img.to(dtype)).view(b, -1, n, d)
        k_ip = self.norm_k_ip(self.k_ip(ip_hidden_states.to(dtype))).view(b, -1, n, d)
        v_ip = self.v_ip(ip_hidden_states.to(dtype)).view(b, -1, n, d)

        k_cloth_vae = self.norm_k(self.k(cloth_latents)).view(b, -1, n, d) + self.norm_k(self.adapter_cloth(cloth_latents)).view(b, -1, n, d)
        v_cloth_vae = self.v(cloth_latents).view(b, -1, n, d) + self.adapter_cloth(cloth_latents).view(b, -1, n, d)

        # img_x = attention(
        #     q.to(dtype), 
        #     k_img.to(dtype), 
        #     v_img.to(dtype), 
        #     k_lens=None
        # )
        # img_x = img_x.to(dtype)
        # compute attention
        x = attention(
            q.to(dtype), 
            k.to(dtype), 
            v.to(dtype), 
            k_lens=context_lens
        )

        ip_x = attention(
            q.to(dtype),
            k_ip.to(dtype),
            v_ip.to(dtype),
            k_lens=None
        )

        img_x_vae = attention(
            q, 
            k_cloth_vae, 
            v_cloth_vae, 
            k_lens=None
        )
        # output
        x = x.to(dtype)
        ip_x = ip_x.to(dtype)
        img_x_vae = img_x_vae.to(dtype)

        x = x.flatten(2)
        # img_x = img_x.flatten(2)
        ip_x = ip_x.flatten(2)
        # x = x + img_x + ip_x
        img_x_vae = img_x_vae.flatten(2)
        x = x + ip_x + img_x_vae
        x = self.o(x)
        return x

import torch.nn as nn
import torch
import math

from diffusers.models.transformers.transformer_2d import BasicTransformerBlock
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from timm.models.vision_transformer import Mlp


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

    
def reshape_tensor(x, heads):
    bs, length, width = x.shape
    #(bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents, shift=None, scale=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class ReshapeExpandToken(nn.Module):
    def __init__(self, expand_token, token_dim):
        super().__init__()
        self.expand_token = expand_token
        self.token_dim = token_dim

    def forward(self, x):
        x = x.reshape(-1, self.expand_token, self.token_dim)
        return x


class TimeResampler(nn.Module):
        # resampler
        # dim=1280,
        # depth=4,
        # dim_head=64,
        # heads=20,
        # num_queries=1024,
        # embedding_dim=1152 + 1536,
        # output_dim=4096,
        # ff_mult=4,
        # timestep_in_dim=320,
        # timestep_flip_sin_to_cos=True,
        # timestep_freq_shift=0,
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
        expand_token=None,
        extra_dim=None,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.expand_token = expand_token is not None
        if expand_token:
            self.expand_proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(embedding_dim * 2, embedding_dim * expand_token),
                ReshapeExpandToken(expand_token, embedding_dim),
                WanRMSNorm(embedding_dim, eps=1e-8),
            )

        self.proj_in = nn.Linear(embedding_dim, dim)
        
        self.extra_feature = extra_dim is not None
        if self.extra_feature:
            self.proj_in_norm = WanRMSNorm(dim, eps=1e-8)
            self.extra_proj_in = torch.nn.Sequential(
                nn.Linear(extra_dim, dim),
                WanRMSNorm(dim, eps=1e-8),
            )

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # msa
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # ff
                        FeedForward(dim=dim, mult=ff_mult),
                        # adaLN
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True))
                    ]
                )
            )

        # time
        self.time_proj = Timesteps(timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift)
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")


    def forward(self, x, timestep, need_temb=False, extra_feature=None):
        timestep_emb = self.embedding_time(x, timestep)  # bs, dim

        latents = self.latents.repeat(x.size(0), 1, 1)
        
        if self.expand_token:
            x = self.expand_proj(x)

        x = self.proj_in(x) # 2688  - (bs,sl,1280)

        if self.extra_feature:
            extra_feature = self.extra_proj_in(extra_feature)
            x = self.proj_in_norm(x)
            x = torch.cat([x, extra_feature], dim=1)
            
        x = x + timestep_emb[:, None] # 注入时间步信息

        for attn, ff, adaLN_modulation in self.layers:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(timestep_emb).chunk(4, dim=1)
            latents = attn(x, latents, shift_msa, scale_msa) + latents
            #  attn:
            #  q: latents
            #  kv: concat(x, latents)
            res = latents
            for idx_ff in range(len(ff)):
                layer_ff = ff[idx_ff]
                latents = layer_ff(latents)
                if idx_ff == 0 and isinstance(layer_ff, nn.LayerNorm):  # adaLN
                    latents = latents * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
            latents = latents + res

            # latents = ff(latents) + latents
            
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        if need_temb:
            return latents, timestep_emb # latent：bs, num_queries, output_dim
        else:                            # timestep_emb: bs, dim
            return latents 


    def embedding_time(self, sample, timestep):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, None)
        return emb


class CrossLayerCrossScaleProjector(nn.Module):
    def __init__(
        self,
        inner_dim=2688,
        num_attention_heads=42,
        attention_head_dim=64,
        cross_attention_dim=2688,
        num_layers=4,

        # resampler
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=1024,
        embedding_dim=1152 + 1536,
        output_dim=4096,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
    ):
        super().__init__()

        self.cross_layer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=0,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="geglu",
                    num_embeds_ada_norm=None,
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type='layer_norm',
                    norm_elementwise_affine=True,
                    norm_eps=1e-6,
                    attention_type="default",
                )
                for _ in range(num_layers)
            ]
        )

        self.cross_scale_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=0,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn="geglu",
                    num_embeds_ada_norm=None,
                    attention_bias=False,
                    only_cross_attention=False,
                    double_self_attention=False,
                    upcast_attention=False,
                    norm_type='layer_norm',
                    norm_elementwise_affine=True,
                    norm_eps=1e-6,
                    attention_type="default",
                )
                for _ in range(num_layers)
            ]
        )

        self.proj = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.proj_cross_layer = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.proj_cross_scale = Mlp(
            in_features=inner_dim, 
            hidden_features=int(inner_dim*2), 
            act_layer=lambda: nn.GELU(approximate="tanh"), 
            drop=0
        )

        self.resampler = TimeResampler(
            dim=dim, #1280
            depth=depth, # 4
            dim_head=dim_head, #64
            heads=heads, # 20
            num_queries=num_queries,# 1024
            embedding_dim=embedding_dim, #1152 + 1536
            output_dim=output_dim, #4096
            ff_mult=ff_mult,    #4
            timestep_in_dim=timestep_in_dim, #320
            timestep_flip_sin_to_cos=timestep_flip_sin_to_cos, #True
            timestep_freq_shift=timestep_freq_shift, #0
        )

    def forward(self, low_res_shallow, low_res_deep, high_res_deep, timesteps, cross_attention_kwargs=None, need_temb=True):
        '''
            low_res_shallow [bs, 729*l, c]
            low_res_deep    [bs, 729, c]
            high_res_deep   [bs, 729*4, c]
        '''

        cross_layer_hidden_states = low_res_deep
        for block in self.cross_layer_blocks:
            cross_layer_hidden_states = block(
                cross_layer_hidden_states, # Query: [B, 729, 2688] 
                encoder_hidden_states=low_res_shallow,# Key/Value: [B, 2187, 2688]
                cross_attention_kwargs=cross_attention_kwargs,
            )
        cross_layer_hidden_states = self.proj_cross_layer(cross_layer_hidden_states)

        cross_scale_hidden_states = low_res_deep
        for block in self.cross_scale_blocks:
            cross_scale_hidden_states = block(
                cross_scale_hidden_states,
                encoder_hidden_states=high_res_deep,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        cross_scale_hidden_states = self.proj_cross_scale(cross_scale_hidden_states)
        
        hidden_states = self.proj(low_res_deep) + cross_scale_hidden_states
        hidden_states = torch.cat([hidden_states, cross_layer_hidden_states], dim=1) 
        # B sl 2688

        hidden_states, timestep_emb = self.resampler(hidden_states, timesteps, need_temb=True)
        return hidden_states, timestep_emb

