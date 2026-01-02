#!/bin/bash

# 1. 设置路径
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_NAME="/data/code/models/Wan-AI/Wan2.2-TI2V-5B"
export DATASET_NAME="/data/code/data/mydata/"

# 【关键】指向新生成的带有宽高信息的 jsonl
export TRAIN_DATASET_META_NAME="/data/code/data/mydata/train_metadata.jsonl"
export TEST_DATASET_META_NAME="/data/code/data/mydata/test_metadata.jsonl"

# 定义输出目录变量，防止手写出错
export OUTPUT_DIR="output_wan_2_2_49frames_1_2"

# 2. 预先创建输出目录 (防止 tee 报错)
mkdir -p $OUTPUT_DIR

# 3. 分布式设置
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1  # 如果你不是多机训练，把 InfiniBand 也关了防止干扰


# 4. 启动训练
echo "Starting training... Logs will be saved to $OUTPUT_DIR/training_output.log"

accelerate launch --config_file config/accelerate_config.yaml \
  scripts/train_dit/train_mymodel.py \
  --config_path="config/train_mymodel/train_wan_2-2.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATASET_NAME \
  --train_data_meta=$TRAIN_DATASET_META_NAME \
  --output_dir=$OUTPUT_DIR \
  --seed=42 \
  --train_batch_size=16 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=4 \
  --num_train_epochs=1000 \
  --checkpointing_steps=500 \
  --max_train_steps=20000 \
  --learning_rate=5e-5\
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --gradient_checkpointing \
  --max_grad_norm=0.1 \
  --max_res=512 \
  --filter_type="image" \
  --video_sample_n_frames=1 \
  --video_sample_stride=1 \
  --rank=128 \
  --network_alpha=64 \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --test_data_meta=$TEST_DATASET_META_NAME \
  --test_steps=2 \
  --test_height=512 \
  --test_width=384 \
  --report_to="wandb" \
  --image_encoder_path="/data/code/models/google/siglip-so400m-patch14-384" \
  --image_encoder_2_path="/data/code/models/facebook/dinov2-giant" \
  2>&1 | tee "$OUTPUT_DIR/training_output.log"