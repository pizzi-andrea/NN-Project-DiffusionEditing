#!/bin/bash

# note
# - utilizzare modello compatibili con la libreria diffusers con struttura [model_name]/vae/ e [model_name]/unet/ 
accelerate launch train_xl/train.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --num_train_epochs=5 \
    --dataset_path=toyset \
    --num_encoders=2 \
    --output_dir=instruct_pix2pix_sd_turbo \
    --resolution=512 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --mixed_precision=bf16 \
    --val_image_url_or_path=apple2.jpg \
    --validation_prompt="make it a green apple" \
    --seed=42 \
