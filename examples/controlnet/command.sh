export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
export CUDA_HOME=/usr/local/cuda/
# accelerate launch train_controlnet.py \export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"

# Training Configuration
accelerate launch --deepspeed_config_file ds_config.json train_controlnet.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir=fusing/fill50k \
  --no_captions \
  --conditioning_image_column="conditioning_image" \
  --image_column="image" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=8 \
  --set_grads_to_none \
  --num_train_epochs=5 \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --output_dir="/root/shreyas/output" \
  --checkpointing_steps=100 \
  --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
  --validation_prompt "" \
  --seed=42 \
  --dataloader_num_workers=64 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  # --use_8bit_adam
  #  --mixed_precision fp16

