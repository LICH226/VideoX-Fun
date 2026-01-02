#!/bin/bash

# 添加当前目录到 PYTHONPATH，防止找不到 videox_fun
export PYTHONPATH=$PWD:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=4,5,6,7

# --- 路径配置 (与训练一致) ---
export MODEL_NAME="/data/code/models/Wan-AI/Wan2.2-TI2V-5B"
export TEST_DATA_DIR="/data/code/data/mydata/"
export TEST_META="/data/code/data/mydata/test_metadata.jsonl"

export SIGLIP_PATH="/data/code/models/google/siglip-so400m-patch14-384"
export DINOV2_PATH="/data/code/models/facebook/dinov2-giant"

# --- LoRA 配置 ---
export LORA_CKPT="output_wan_2_2_49frames_12_31/checkpoint-10000"
export INFER_OUTPUT="eval_results/checkpoint_10000"

echo "------------------------------------------------"
echo "Starting Inference with Strict Loading Logic..."
echo "Base Model: $MODEL_NAME"
echo "LoRA Path:  $LORA_CKPT"
echo "------------------------------------------------"

python scripts/train_dit/test_pipeline.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --lora_checkpoint_path="$LORA_CKPT" \
  --config_path="config/train_mymodel/train_wan_2-2.yaml" \
  --image_encoder_path="$SIGLIP_PATH" \
  --image_encoder_2_path="$DINOV2_PATH" \
  --test_data_dir="$TEST_DATA_DIR" \
  --test_data_meta="$TEST_META" \
  --output_dir="$INFER_OUTPUT" \
  --height=512 \
  --width=384 \
  --num_inference_steps=50 \
  --guidance_scale=5.0 \
  --seed=42 \
  --num_samples=20