from pathlib import Path
from venv import logger
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

from torch.utils.data import DataLoader

class Pix2Pix_DatasetFactory:
    def __init__(self, dataset_dir:str|Path, src_prompts_path:str|Path, llm_weights:str|Path, clip_weights:str|Path|None=None) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
                
        self.logger = logging.getLogger(__name__)
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
    

    def generate_pair_images(self, prompts:list[tuple[str,str]]|str|Path|DatasetDict, original_prompt:str="original_prompt", edited_prompt:str="edited_prompt", device:str='cpu',  save_name:str|None = None) -> list[tuple[PIL.Image, str, PIL.Image]]:
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


    def generate_new_dataset(self, num_samples:int, columns_prompt:str="original_prompt", save_mid_steps:bool=False, device:str='cpu', batch_s:int=8) -> DatasetDict:
        self.new_dataset:DatasetDict|None = None
        edit_name = None 
        
        if save_mid_steps:
            edit_name = "edited_prompts"
        edited_prompts = self.generate_edit_prompts(num_samples, columns_prompt, device, batch_s, edit_name)
        self.generate_pair_images(edited_prompts, device=device, save_name=edit_name)

        return self.new_dataset # type: ignore


if __name__ == '__main__':
    ds_factory = Pix2Pix_DatasetFactory("dataset/myDataset", "dataset/spixset/test", "weights/mistral")
    
        

