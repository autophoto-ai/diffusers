#!/bin/bash

# Example script to train ControlNet with LoRA
# This demonstrates the new LoRA functionality added to the training script

# Set environment variables
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/root/shreyas/output_fusing_fill50k_lora"
export DATASET_DIR="fusing/fill50k"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with LoRA enabled
python /root/shreyas/diffusers/examples/controlnet/train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_data_dir=$DATASET_DIR \
  --use_lora \
  --lora_rank=16 \
  --lora_alpha=32 \
  --lora_dropout=0.1 \
  --lora_target_modules "to_k" "to_q" "to_v" "to_out.0" \
  --resolution=512 \
  --learning_rate=1e-4 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=51000 \
  --checkpointing_steps=500 \
  --seed=42 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  # --validation_steps=100 \
  # --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
  # --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
  # --report_to="tensorboard"

echo "LoRA ControlNet training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "LoRA adapter size is much smaller than full model (~10-100MB vs 2-7GB)"
