import PIL 
import numpy as np
import os
import torch
from contextlib import nullcontext
from transformers import PretrainedConfig
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import load_image
from diffusers import UNet2DConditionModel


from diffusers.training_utils import EMAModel
def convert_to_np(image, resolution):
    if isinstance(image, str):
        image = PIL.Image.open(image)
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def save_model_hook(accelerator,models, weights, output_dir, ema=None):
        if accelerator.is_main_process:
            if ema:
                ema.save_pretrained(output_dir.joinpath("unet_ema"))

            for _, model in enumerate(models):
                model.save_pretrained(output_dir.joinpath("unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()


def unwrap_model(model, accelerator):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

 # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format

def load_model_hook(accelerator, models, input_dir, ema=None):
    if ema:
        load_model = EMAModel.from_pretrained(input_dir.joinpath("unet_ema"), UNet2DConditionModel)
        ema.load_state_dict(load_model.state_dict())
        ema.to(accelerator.device)
        del load_model

    for i in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model

# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

def preprocess_images(examples, resolution, original_image_column, edited_image_column, transforms_callable=None):
        original_images = np.concatenate(
            [convert_to_np(image, resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, resolution) for image in examples[edited_image_column]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return transforms_callable(images) if transforms_callable else images

def log_validation(logger, image_path, validation_prompt, out_dir, pipeline, num_validation_images, accelerator, generator, global_step):
    """Generate validation sample on single imageusing given pipline (trained model + preprocessing operations)"""
    logger.info(
        f"Running validation... \n Generating {num_validation_images} images with prompt:"
        f" {validation_prompt}."
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    val_save_dir = out_dir.joinpath("validation_images")
    if not val_save_dir.exists():
        os.makedirs(val_save_dir)

    original_image = load_image(image_path)
      

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        edited_images = []
        # Run inference
        for val_img_idx in range(num_validation_images): # attempts generation 
            a_val_img = pipeline(
                validation_prompt,
                image=original_image,
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator,
            ).images[0]
            edited_images.append(a_val_img)
            # Save validation images
            a_val_img.save(os.path.join(val_save_dir, f"step_{global_step}_val_img_{val_img_idx}.png"))

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_xl(text_encoders, tokenizers, prompt, logger=None):
    prompt_embeds_list = []

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
            if logger:
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )
            else:
                print("The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}")

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # We are only ALWAYS interested in the pooled output of the final text encoder
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds

# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompts_xl(text_encoders, tokenizers, prompts):
    prompt_embeds_all = []
    pooled_prompt_embeds_all = []

    for prompt in prompts:
        prompt_embeds, pooled_prompt_embeds = encode_prompt_xl(text_encoders, tokenizers, prompt)
        prompt_embeds_all.append(prompt_embeds)
        pooled_prompt_embeds_all.append(pooled_prompt_embeds)

    return torch.stack(prompt_embeds_all), torch.stack(pooled_prompt_embeds_all)

# Adapted from examples.dreambooth.train_dreambooth_lora_sdxl
# Here, we compute not just the text embeddings but also the additional embeddings
# needed for the SD XL UNet to operate.
def compute_embeddings_for_prompts_xl(accelerator, prompts, text_encoders, tokenizers):
    with torch.no_grad():
        prompt_embeds_all, pooled_prompt_embeds_all = encode_prompts_xl(text_encoders, tokenizers, prompts)
        add_text_embeds_all = pooled_prompt_embeds_all

        prompt_embeds_all = prompt_embeds_all.to(accelerator.device)
        add_text_embeds_all = add_text_embeds_all.to(accelerator.device)
    return prompt_embeds_all, add_text_embeds_all


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
        captions = list(examples[edit_prompt_column])
        prompt_embeds_all, add_text_embeds_all = compute_embeddings_for_prompts_xl(accelerator, captions, text_encoders, tokenizers)
        examples["prompt_embeds"] = prompt_embeds_all
        examples["add_text_embeds"] = add_text_embeds_all
        return examples