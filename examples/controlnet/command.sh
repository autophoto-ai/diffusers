# export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"
# export CUDA_HOME=/usr
# accelerate launch train_controlnet.py \export MODEL_DIR="stable-diffusion-v1-5/stable-diffusion-v1-5"

# Training Configuration
accelerate launch --deepspeed_config_file ds_config.json train_controlnet.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="/home/ml3au/Desktop/Experiments/controlnet-shadow/data/data" \
  --no_captions \
  --conditioning_image_column="input_small" \
  --image_column="targets_small" \
  --resolution=256 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --set_grads_to_none \
  --num_train_epochs=5 \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --output_dir="/home/ml3au/Desktop/Experiments/controlnet-shadow/data/data/output" \
  --checkpointing_steps=100 \
  --validation_image "./U4Xfpgt8cBxyUdE69hxcP6fyoofziLSSpuEduRlPW0kpPMaeaUi2fY8qkwkTiyg620240920145329.png" "./UrqcLVT3Rb6XCTMmBc0IkoURCTo13KK8cZ2fNPdl1WydaBP1WboE7VsHMM4gYBpJ20240920145110.png" \
  --validation_prompt "" \
  --seed=42 \
  --dataloader_num_workers=2 \
  --adam_beta1=0.9 \
  --adam_beta2=0.999 \
  --adam_weight_decay=1e-2 \
  --adam_epsilon=1e-08 \
  --use_8bit_adam
# #  --mixed_precision fp16

