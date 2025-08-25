#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Harutatsu Akiyama and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import math
import os
import shutil
import warnings
import numpy as np
from pathlib import Path



import datasets
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import BitsAndBytesConfig
from torchvision import transforms
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import diffusers

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline

from diffusers.training_utils import EMAModel
from train_utils import compute_embeddings_for_prompts, compute_null_conditioning, load_model_hook, log_validation, preprocess_images, unwrap_model, instance_txt_encoder, CLIP_Score
from args_parser import parse_args
from datasets import load_from_disk

import matplotlib
matplotlib.use("Agg")  # backend non interattivo
logger = get_logger(__name__, log_level="INFO")
use_ema = False

file_path = "dataset_config.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# riconverti le liste in tuple
DATASET_NAME_MAPPING = {k: tuple(v) for k, v in data.items()}

TORCH_DTYPE_MAPPING = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def plot_loss(loss_history: list[float], save_path: str|Path):
    """
    
    Args:
        loss_history (list[float]): Lista dei valori di loss per epoca.
        save_path (str): Cartella in cui salvare il plot.
    """
    path = Path(save_path)
    
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', color='blue')
    plt.title("Loss History")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    
    file_path = path.joinpath("loss_history.png")
    plt.savefig(file_path)
    plt.close()


def train():

    
    args = parse_args()

    # log configuration
    out_dir = Path(args.output_dir)
    log_dir = Path(args.logging_dir)
    model_name = args.pretrained_model_name_or_path.split("/")[-1]
    dataset_path = Path(args.dataset_path)
    logging_dir = out_dir.joinpath(log_dir)

   
    # accelerator configuration
    accelerator_project_config = ProjectConfiguration(project_dir=out_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_config=accelerator_project_config,
    )

    

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    
    if not out_dir.exists():
        os.makedirs(out_dir, exist_ok=True)
    else:
        logger.info(f"directory '{out_dir}' exist and may be not empty")

    # end log configuration

    
    
    # Variation Auto Encoder configuration

    quantization_config = None 
    if args.quantization:
        args.mixed_precision = "bf16"
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # Tipo di quantizzazione, "nf4" è il più comune
        bnb_4bit_use_double_quant=True, # Abilita la doppia quantizzazione per maggiore precisione
        bnb_4bit_compute_dtype=torch.bfloat16, # Tipo di dato per i calcoli intermedi
        )
        logger.info("enable pretraned-weights quantization")

    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        quantization_config=quantization_config,
    )

    # Unet folder

    unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet",
            torch_dtype=TORCH_DTYPE_MAPPING[args.mixed_precision]
        )
    
    
    



    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet.
    
    #  This UNet is then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    logger.info(f"Initializing the {model_name} UNet from the pretrained UNet.")
    in_channels = 8 # 4 channels for input-image + 4 channels for guidance-image
    out_channels = unet.conv_in.out_channels
    unet.register_to_config(in_channels=in_channels)

    # new convolutional layer
    with torch.no_grad():
        new_conv_in = nn.Conv2d(
            in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
        )
        new_conv_in.weight.zero_()
        new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        unet.conv_in = new_conv_in


    if args.lora:
        lora_config = LoraConfig(
            r = 32,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["to_q","to_k","to_v","to_out.0"],
            inference_mode=False
        )

        unet = get_peft_model(unet, lora_config)
        #unet.add_adapter(lora_config, adapter_name='quantizator')
        tr, p = unet.get_nb_trainable_parameters()
        logger.info("number of trainable parameters: %d|%d (%f)", tr,p, 100*(tr/p))
        
    # Create EMA for the unet.
    if use_ema:
        ema = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)


    def save_model_hook(models, weights, output_dir):
            if use_ema:
                ema.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()
    
    

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    
    # initialize Adam optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # end adam optimizer
 
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.

    if not dataset_path.exists():
        FileNotFoundError(f"dataset not found in {dataset_path}")
    

    dataset = load_from_disk(dataset_path)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if isinstance(dataset, datasets.Dataset):
        dataset = datasets.DatasetDict({
            "train":dataset
        })
    
    
    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(dataset_path.name, None)
    if not dataset_columns:
        raise ValueError("Unknow dataset schema")
    
    original_image_column = dataset_columns[0] 
    edit_prompt_column = dataset_columns[1] 
    edited_image_column = dataset_columns[2] 
   

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        warnings.warn(f"weight_dtype {weight_dtype} may cause nan during vae encoding", UserWarning)

    elif accelerator.mixed_precision == "bf16" or args.quantization:
        weight_dtype = torch.bfloat16
        warnings.warn(f"weight_dtype {weight_dtype} may cause nan during vae encoding", UserWarning)

    

    # Preprocessing the datasets.
    train_transforms = transforms.Compose([
                transforms.Lambda(
                    lambda x: x
                )
            ]
        )
    if args.preprocessing:
        train_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(args.resolution)
            ]
        )

    
    # load tokenizer for conditional prompt
    # note: each diffusion model use one or more text encoder in order to ensure a better understanding of the prompt

    

    # Load scheduler and models (update with dynamic import)
    scheduler_conf = DDPMScheduler.load_config(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )

    scheduler_cls = scheduler_conf["_class_name"]
    logger.info("scheduler dected %s", scheduler_cls)
    
    if scheduler_cls == "DDIMScheduler":
        noise_scheduler_cls = DDPMScheduler
    elif scheduler_cls == "PNDMScheduler":
        from diffusers.schedulers.scheduling_pndm import PNDMScheduler
        noise_scheduler_cls = PNDMScheduler
    elif scheduler_cls == "EulerDiscreteScheduler":
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
        noise_scheduler_cls = EulerDiscreteScheduler
    elif scheduler_cls == "EulerAncestralDiscreteScheduler":
        from diffusers.schedulers.scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
        noise_scheduler_cls = EulerAncestralDiscreteScheduler
    else:
        raise ValueError(f"impossible detectd class cheduler {scheduler_cls}")
    
    
    logger.info("scheduler instance %s", noise_scheduler_cls.__name__)
    noise_scheduler = noise_scheduler_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    tokenizers, text_encoders = instance_txt_encoder(args.pretrained_model_name_or_path, device=accelerator.device, num_encoders=args.num_encoders, dtype=weight_dtype, quantization_config=quantization_config)
    

    # Freeze vae and text_encoders
    vae.requires_grad_(False)
    for i in range(len(text_encoders)):
        text_encoders[i].requires_grad_(False)
########

    def compute_time_ids():
        crops_coords_top_left = (0, 0)
        original_size = target_size = (args.resolution, args.resolution)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype)
        return add_time_ids.to(accelerator.device).repeat(args.train_batch_size, 1)

    add_time_ids = compute_time_ids()

    def _preprocess_train(accelerator, examples, resolution, original_image_column, edited_image_column, edit_prompt_column, text_encoders, tokenizers, callable_for_images=None):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples, resolution, original_image_column, edited_image_column, callable_for_images)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, resolution, resolution)
        edited_images = edited_images.reshape(-1, 3, resolution, resolution)

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        # Preprocess the captions.
        captions = list(examples[edit_prompt_column]) # get captions
        prompt_embeds_all, add_text_embeds_all = compute_embeddings_for_prompts(accelerator, captions, text_encoders, tokenizers) # embeddings
        examples["prompt_embeds"] = prompt_embeds_all
        examples["add_text_embeds"] = add_text_embeds_all
        return examples
    
    def preprocess_train(examples):
        return _preprocess_train(accelerator, examples, args.resolution, original_image_column, edited_image_column, edit_prompt_column, text_encoders, tokenizers, train_transforms)

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
   
    
    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)
    
    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        prompt_embeds = torch.concat([example["prompt_embeds"] for example in examples], dim=0)
        #add_text_embeds = torch.concat([example["add_text_embeds"] for example in examples], dim=0)
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "prompt_embeds": prompt_embeds,
            #"add_text_embeds": add_text_embeds,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if use_ema:
        ema.to(accelerator.device)

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
   
    vae.to(accelerator.device)
        
    
    
    unet.to(accelerator.device, dtype=weight_dtype)
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    
    accelerator.init_trackers(model_name, config=vars(args))
    clip_score = CLIP_Score("openai/clip-vit-base-patch32")
    
    # outload text encoder and txt encoder order to free GPU memory
    
    
    #for encoder_txt in text_encoders:
    #    encoder_txt.to('cpu')

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    val = False
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(out_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(out_dir.joinpath(path)) # TODO: need fix some problems in callback hook
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    history_loss = []
    epoch_loss = []
    mean_epoch = []
   
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        logger.info("start epoch %d", epoch)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
               
                edited_pixel_values = batch["edited_pixel_values"].to(vae.dtype)
                latents = vae.encode(edited_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL additional inputs
                encoder_hidden_states = batch["prompt_embeds"]
                #add_text_embeds = batch["add_text_embeds"]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.

                original_pixel_values = batch["original_pixel_values"].to(weight_dtype)
                #original_image_embeds = vae.encode(original_pixel_values.to(torch.float32)).latent_dist.sample()
                original_image_embeds = vae.encode(original_pixel_values.to(torch.float32)).latent_dist.mode()
                original_image_embeds = original_image_embeds.to(weight_dtype)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://huggingface.co/papers/2211.09800.
                
                
                if args.conditioning_dropout_prob is not None:
                    #logger.info("conditional dropout enabled")
                    null_conditioning = compute_null_conditioning(tokenizers, text_encoders, accelerator)
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.

                
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                
                added_cond_kwargs = {"time_ids": add_time_ids}
                #added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
               
               
                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,

                )[0]
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = loss.mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if use_ema:
                    ema.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                val = True # enable validation when update global_step is update 
                accelerator.log({"train_loss": train_loss}, step=global_step)
                history_loss.append(train_loss)
                epoch_loss.append(train_loss)
                train_loss = 0.0

                if global_step % round(len(train_dataset)/(total_batch_size)) == 0:
                    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    
                    checkpoints = os.listdir(out_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= 3:
                        num_to_remove = len(checkpoints) - 3 + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(out_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(out_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    plot_loss(history_loss, Path(save_path))
                    logger.info(f"Saved state to {save_path}")
                    
                    

            
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            

            ### BEGIN: Perform validation every `validation_epochs` steps
            if global_step % round(len(train_dataset)/(total_batch_size)) == 0 and args.validation_steps != 0 and val:
                unet.eval()
                logger.info("Perform validation %d steps", global_step)
                if (args.val_image_url_or_path is not None) and (args.validation_prompt is not None):
                    # create pipeline
                    if use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema.store(unet.parameters())
                        ema.copy_to(unet.parameters())

                    # The models need unwrapping because for compatibility in distributed training mode.
                    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet, accelerator),
                        text_encoder=text_encoders[0],
                        #text_encoder_2=text_encoders[1],
                        tokenizer=tokenizers[0],
                        #tokenizer_2=tokenizers[1],
                        vae=vae,
                        torch_dtype=weight_dtype,
                        safety_checker=None
                    )
                    #pipeline.enable_model_cpu_offload()
                    
                    log_validation(
                        logger,
                        args.val_image_url_or_path,
                        args.validation_prompt,
                        out_dir,
                        pipeline,
                        args.num_validation_images,
                        accelerator,
                        generator,
                        global_step,
                        clip_score
                    )

                    

                    if use_ema:
                        # Switch back to the original UNet parameters.
                        ema.restore(unet.parameters())

                    del pipeline
                    torch.cuda.empty_cache()
                # disable validation untill next global_step not occure
                val = False
                unet.train()
            ### END: Perform validation every `validation_epochs` steps
        # end step
        
        mean_epoch.append(np.mean(epoch_loss).item())
        plot_loss(mean_epoch, out_dir)
        logger.info("end epoch %d with mean loss=%f", epoch, mean_epoch[-1])
        epoch_loss.clear()
    # end epoch
   
   
   
    # Create the pipeline using the trained modules and save it.
        

    if use_ema:
        ema.copy_to(unet.parameters())

    unet.eval()
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unwrap_model(unet, accelerator),
        text_encoder=text_encoders[0],
        #text_encoder_2=text_encoders[1],
        tokenizer=tokenizers[0],
        #tokenizer_2=tokenizers[1],
        vae=vae,
        torch_dtype=weight_dtype,
        safety_checker=None
    )

    pipeline.save_pretrained(out_dir)
    #pipeline.enable_model_cpu_offload()

    if (args.val_image_url_or_path is not None) and (args.validation_prompt is not None):
        log_validation(
                    logger,
                    args.val_image_url_or_path,
                    args.validation_prompt,
                    out_dir,
                    pipeline,
                    args.num_validation_images,
                    accelerator,
                    generator,
                    global_step,
                    clip_score
                )

    accelerator.end_training()

if __name__ == "__main__":
    train()