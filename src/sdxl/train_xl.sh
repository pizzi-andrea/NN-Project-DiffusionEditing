#!/bin/bash

# note
# - utilizzare modello compatibili con la libreria diffusers con struttura [model_name]/vae/ e [model_name]/unet/ 
accelerate launch train/train_xl.py \
    --pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0 \
    --dataset_path=toyset \
    --resolution=256 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=15000 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --mixed_precision=fp16 \
    --seed=42
