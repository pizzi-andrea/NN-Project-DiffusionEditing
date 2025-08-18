from pathlib import Path

from venv import logger
import random
import PIL.Image
import torch
import logging
import torch
import PIL
import os
from datasets import DatasetDict
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from torch.utils.data import DataLoader
from .promptToPrompt import PromptToPromptGenerator
from datasets import Dataset, Features, Value
from datasets import Dataset, Features, Value, Image as HFDatasetImage

from PIL import Image, ImageDraw, ImageFont
def save_pair_with_label(img1, img2, prompt, save_path):
    # affianca le due immagini orizzontalmente
    w, h = img1.size
    combined = Image.new("RGB", (w * 2, h), color=(255, 255, 255))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (w, 0))

    # disegna il prompt in basso
    draw = ImageDraw.Draw(combined)

    # scegli un font (se non trovato, usa default)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default(20)

    # bounding box del testo (xmin, ymin, xmax, ymax)
    bbox = draw.textbbox((0, 0), prompt, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # posizione centrata in basso
    x = (combined.width - text_w) // 2
    y = h - text_h - 5  # 5px dal bordo in basso

    # rettangolo nero dietro al testo
    draw.rectangle([(x-5, y-2), (x+text_w+5, y+text_h+2)], fill=(0,0,0,127))
    draw.text((x, y), prompt, font=font, fill=(255,255,255))

    combined.save(save_path)

import numpy as np

def convert_to_np(image, resolution):
    # converto in RGB e ridimensiono
    image = image.convert("RGB").resize(resolution)
    return np.array(image)   # shape: (H, W, C) -> compatibile con datasets.Image

class Pix2Pix_DatasetFactory:
    def __init__(self, dataset_dir:str|Path, src_prompts_path:str|Path, llm_weights:str|Path, diffusion_id:str, clip_weights:str|Path|None=None) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
                
        self.logger = logging.getLogger(__name__)
        self.diffusion_id = diffusion_id
        self.dataset = Path(src_prompts_path)
        self.base_path = Path(dataset_dir)
        self.llm_weights = Path(llm_weights)
        self.clip_weights = Path(clip_weights) if clip_weights else None


        if not self.base_path.exists():
            os.makedirs(self.base_path, exist_ok=True)
            logger.info("create dataset directory with path:%s", str(self.base_path.absolute()))
        else:
            logger.warning(f"directory:%s just exist", str(self.base_path))

        # sanity check

        if not (self.dataset.exists() and self.llm_weights.exists() ):
            raise FileNotFoundError("some specified path not exist, check if paths exits")
        
    def generate_edit_prompts(self, num_samples:int, columns_prompt:str="original_prompt",device:str='cpu', batch_s:int=8, save_name:str|None = None) -> list[tuple[str,str]]:
        weights_type = torch.bfloat16 if torch.cuda.is_available() and device=='cuda' else torch.float16

        
        
        dataset = load_from_disk(self.dataset).shuffle().select(range(0, num_samples))
        if not columns_prompt in dataset.column_names:
            dataset = dataset.rename_column(columns_prompt, "original_prompt")
        self.logger.info("load dataset, num sample %d", dataset.num_rows)
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  # or torch.float16 if bfloat16 not supported
            )
        llm = AutoModelForCausalLM.from_pretrained(self.llm_weights, 
            torch_dtype=weights_type, 
            device_map=device, 
            quantization_config=bnb_config)
        outputs = []
        tokenizer = AutoTokenizer.from_pretrained(self.llm_weights)
        
        llm.eval()

        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch]),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch])
        }

        def preprocess(example):
            prompt = example["original_prompt"] + "\n##\n"
            inputs = tokenizer(\
                prompt,
                truncation=True,
                padding="max_length",
                max_length=180
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
        }

        self.logger.info("load llm weights")
        tk_dataset = dataset.map(preprocess)
        ds_loader = DataLoader(tk_dataset, batch_size=batch_s, shuffle=False, collate_fn=collate_fn)
        self.logger.info("starting generation...")
        for batch in tqdm(ds_loader, desc="Generating prompts", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            generation = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                pad_token_id=tokenizer.eos_token_id
            )
            
            outputs.extend(tokenizer.batch_decode(generation, skip_special_tokens=True))

        edit_prompt = []
        edited_prompt = []
        
        for idx, prompt in enumerate(outputs):
            try:
                edit_prompt.append(prompt.split("\n##\n")[1].split("\n%%\n")[0])
                
            except IndexError:
                edit_prompt.append("?")
                logger.warning("sample %d will be empty", idx)
            try:
                
                edited_prompt.append(prompt.split("\n%%\n")[1].split("\nEND")[0])
            except IndexError:
                edited_prompt.append("?")
                logger.warning("sample %d will be empty", idx)


            
        logger.info("Generated %d/%d edited_prompt/edit_prompt", len(edited_prompt), len(edit_prompt))
        dataset = dataset.add_column("edited_prompt", edited_prompt)
        dataset = dataset.add_column("edit_prompt",   edit_prompt)
        
        if save_name:
            dataset.save_to_disk(self.base_path.joinpath(save_name))
            self.logger.info("save generations in:%s", str(self.base_path.joinpath(save_name)))
        
        self.new_dataset = dataset
        return [(o, e) for (o, e) in zip(dataset['original_prompt'], edited_prompt)]
    

    def generate_pair_images(self, prompts:list[tuple[str,str]]|str|Path|DatasetDict, original_prompt:str="original_prompt", edited_prompt:str="edited_prompt", steps:int=4, alpha:float=0.7, scale:float=7.5,device:str='cpu', resolution:tuple[int,int]=(512,512), tr:int=3, eval_step:int=2, save_name:str|None = None) -> list[tuple[PIL.Image, str, PIL.Image]]:
        # parse src of prompts
        if isinstance(prompts, (str,Path)):
            dataset = load_from_disk(prompts).select_columns([original_prompt, edited_prompt])
        if isinstance(prompts, DatasetDict):
            pass 
        if isinstance(prompts, list):
            pass
        else:
            ValueError("prompts format not valid")
        
        #TODO:
        # - applicare il modello clip per il filtraggio delle immagini
        # - capire il formato dei dati da dare in input al modello

        
        #dataset = dataset.rename_columns({
        #    original_prompt : "original_prompt",
        #    edited_prompt   : "edited_prompt"
        #})


        pipe = StableDiffusionPipeline.from_pretrained(self.diffusion_id, torch_dtype=torch.bfloat16)        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        p2p = PromptToPromptGenerator(pipe, num_inference_steps=steps, device=device)

        original_imgs = []
        edited_imgs = []
        edit_prompt = []
        bst_img1 = None
        bst_img2 = None
        for idx, (original, edited) in tqdm(enumerate(prompts)):
            max_score = 1
            for i in range(tr):
                torch.cuda.empty_cache()
                p = random.uniform(0.1, 0.9)
                self.logger.info(f"p = %f", p)
                img1, img2, score = p2p.generate(original,
                            edited, 
                            p = p,
                            alpha = alpha,
                            guidance_scale = scale,
                            #path = "./"
                            )

                if score < max_score:
                    #print(f"new max score: {score}")
                    max_score = score
                    bst_img1 = img1
                    bst_img2 = img2
            
            if idx % eval_step == 0:
                save_pair_with_label(bst_img1, bst_img2, f"original_prompt:{original}'\n'edited_prompt:{edited}",self.base_path.joinpath(f"sample-{idx}.png") )
                self.logger.info("log pairs image in:%s", str(self.base_path.absolute()))
            
            original_imgs.append(bst_img1)
            edited_imgs.append(bst_img2)
            edit_prompt.append(edited)
        
        if self.dataset:

            features = Features({
                "original_image": HFDatasetImage(),   # colonne immagine
                "edited_image": HFDatasetImage()
            })

            # costruisci il dataset con le features
            data = Dataset.from_dict({
                "original_image": original_imgs,
                "edited_image": edited_imgs,
            }, features=features)

            tmp = {**self.new_dataset.to_dict(), **data.to_dict()}
            self.new_dataset =  Dataset.from_dict(tmp)
            self.logger.info("Add columns original_image/edited_image to generated dataset")
        
        
        if save_name:
            features = Features({
                "original_image": HFDatasetImage(),   # colonne immagine
                "edit_prompt": Value("string"),
                "edited_image": HFDatasetImage()
            })

            # costruisci il dataset con le features
            data = Dataset.from_dict({
                "original_image": original_imgs,
                "edit_prompt": edit_prompt,
                "edited_image": edited_imgs,
            }, features=features)

            data.save_to_disk(self.base_path.joinpath(save_name))
            self.logger.info("save generations in:%s", str(self.base_path.joinpath(save_name)))
        
        return [(src, prompt, edit) for src, prompt, edit in zip(original_imgs, edit_prompt, edited_imgs)]
                
            



    def generate_new_dataset(self, num_samples:int, columns_prompt:str="original_prompt", save_mid_steps:bool=False, device:str='cpu', batch_s:int=8) -> DatasetDict:
        self.new_dataset:DatasetDict|None = None
        edit_name = None 
        
        if save_mid_steps:
            edit_name = "edited_prompts"
        edited_prompts = self.generate_edit_prompts(num_samples, columns_prompt, device, batch_s, edit_name)
        self.generate_pair_images(edited_prompts, device=device, save_name=edit_name)

        return self.new_dataset # type: ignore


if __name__ == '__main__':
    ds_factory = Pix2Pix_DatasetFactory("dataset/myDataset", "dataset/spixset/test", "weights/mistral", "stabilityai/stable-diffusion-2-base")
    prompts = ds_factory.generate_edit_prompts(10, device='cuda')
    ds_factory.generate_pair_images(prompts=prompts, device='cuda')
        

