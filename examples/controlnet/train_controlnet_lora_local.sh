#!/bin/bash
# ControlNet LoRA Training Example with Your Dataset
# This script demonstrates how to train a ControlNet with LoRA using your local dataset

set -e  # Exit on any error

echo "🚀 Starting ControlNet LoRA Training..."
echo "📁 Using dataset: /root/shreyas/datasets/"
echo "⚡ Training method: LoRA (Low-Rank Adaptation)"

# Navigate to the training directory
cd /root/shreyas/diffusers/examples/controlnet

# Activate the control-net environment (if not already activated)
source activate control-net

# Set training parameters
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATASET_DIR="/root/shreyas/datasets"
OUTPUT_DIR="/root/shreyas/output/controlnet-lora-$(date +%Y%m%d_%H%M%S)"
RESOLUTION=512
BATCH_SIZE=4
EPOCHS=3
LEARNING_RATE=1e-4

echo "📋 Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training with LoRA
python train_controlnet.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$DATASET_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --use_lora \
  --lora_rank=16 \
  --lora_alpha=32 \
  --lora_dropout=0.1 \
  --lora_target_modules "to_k" "to_q" "to_v" "to_out.0" \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --num_train_epochs=$EPOCHS \
  --learning_rate=$LEARNING_RATE \
  --gradient_accumulation_steps=2 \
  --mixed_precision="fp16" \
  --checkpointing_steps=250 \
  --validation_steps=250 \
  --logging_dir="$OUTPUT_DIR/logs" \
  --report_to="tensorboard" \
  --tracker_project_name="controlnet-lora-training" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --dataloader_num_workers=4 \
  --max_grad_norm=1.0 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --seed=42

echo ""
echo "✅ Training completed!"
echo "📁 Model saved to: $OUTPUT_DIR"
echo "📊 View training logs with: tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "💡 Next steps:"
echo "  1. Check the training logs in TensorBoard"
echo "  2. Test your trained LoRA with the inference script"
echo "  3. Experiment with different lora_rank values for better results"
