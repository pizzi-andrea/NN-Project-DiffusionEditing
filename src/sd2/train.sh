#!/bin/bash

# note
# - utilizzare modello compatibili con la libreria diffusers con struttura [model_name]/vae/ e [model_name]/unet/ 
accelerate launch train/train.py \
    --pretrained_model_name_or_path=stabilityai/sd-turbo \
    --num_train_epochs=4 \
    --dataset_path=toyset \
    --num_encoders=1 \
    --output_dir=instruct_pix2pix_sd2 \
    --resolution=256 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --learning_rate=3e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --mixed_precision=bf16 \
    --val_image_url_or_path=apple2.jpg \
    --validation_prompt="make it a green apple" \
    --seed=42 \
