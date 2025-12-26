#!/bin/bash

# 1. 设置路径
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MODEL_NAME="/data/code/models/alibaba-pai/Wan2.1-Fun-14B-InP"
export DATASET_NAME="/data/code/data/mydata/"

# 【关键】指向新生成的带有宽高信息的 jsonl
export TRAIN_DATASET_META_NAME="/data/code/data/mydata/train_metadata_with_hw.jsonl"
export TEST_DATASET_META_NAME="/data/code/data/mydata/test_metadata.jsonl"

# 定义输出目录变量，防止手写出错
export OUTPUT_DIR="output_wan_49frames_12_23"

# 2. 预先创建输出目录 (防止 tee 报错)
mkdir -p $OUTPUT_DIR

# 3. 分布式设置
export NCCL_DEBUG=WARN

# 4. 启动训练
echo "Starting training... Logs will be saved to $OUTPUT_DIR/training_output.log"

accelerate launch --config_file config/accelerate_config.yaml \
  scripts/wan2.1_fun/train_mymodel.py \
  --config_path="config/wan2.1/wan_civitai_mymodel.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$TRAIN_DATASET_META_NAME \
  --output_dir=$OUTPUT_DIR \
  --seed=42 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=16 \
  --dataloader_num_workers=4 \
  --num_train_epochs=1000 \
  --checkpointing_steps=200 \
  --max_train_steps=20000 \
  --learning_rate=1e-5 \
  --resume_from_checkpoint="checkpoint-200" \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --gradient_checkpointing \
  --max_grad_norm=0.05 \
  --video_sample_size=512 \
  --video_sample_n_frames=49 \
  --video_sample_stride=1 \
  --rank=128 \
  --network_alpha=64 \
  --lora_skip_name="ffn" \
  --motion_sub_loss \
  --motion_sub_loss_ratio=0.25 \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --validation_paths=$TEST_DATASET_META_NAME \
  --validation_steps=500 \
  --report_to="tensorboard" \
  --image_encoder_path="/data/code/models/google/siglip-so400m-patch14-384" \
  --image_encoder_2_path="/data/code/models/facebook/dinov2-giant" \
  2>&1 | tee "$OUTPUT_DIR/training_output.log"