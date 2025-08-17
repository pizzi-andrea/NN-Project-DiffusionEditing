from pathlib import Path
from venv import logger
import PIL.Image
import torch
import logging
import torch
import PIL
import numpy as np 


from datasets import DatasetDict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from src.libs.evaluatorFactory import compute_metrics_factory
from torch.utils.data import DataLoader
from mistralLoraTrainer import MistralLoraTrainer

class Pix2Pix_DatasetFactory:
    def __init__(self, dataset_dir:str|Path, src_prompts_path:str|Path, llm_weights:str|Path, clip_weights:str|Path) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
                
        self.logger = logging.getLogger(__name__)
        self.dataset = Path(src_prompts_path)
        self.base_path = Path(dataset_dir)
        self.llm_weights = Path(llm_weights)
        self.clip_weights = Path(clip_weights)


        if not self.base_path.exists():
            logger.info("create dataset directory with path:%s", str(self.base_path))
        else:
            logger.warning(f"directory:%s just exist", str(self.base_path))

        # sanity check

        if not (self.dataset.exists() and self.llm_weights.exists() and self.clip_weights.exists() ):
            raise FileNotFoundError("some specified path not exist, check if paths exits")
        
    def generate_edit_prompts(self, num_samples:int, columns_prompt:str="original_prompt",device:str='cpu', batch_s:int=8, save_name:str|None = None) -> list[str]:
        weights_type = torch.bfloat16 if torch.cuda.is_available() and device=='cuda' else torch.float16

        def collate_fn(batch):
            return {
                "input_ids": torch.tensor([item["input_ids"] for item in batch]),
                "attention_mask": torch.tensor([item["attention_mask"] for item in batch])
        }
        dataset = load_from_disk(self.dataset).shuffle().select(range(0, num_samples))
        dataset = dataset.rename_column(columns_prompt, "original_prompt")
        self.logger.info("load dataset, num sample %d", dataset.num_rows)
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  # or torch.float16 if bfloat16 not supported
            )
        llm = AutoModelForCausalLM.from_pretrained(self.llm_weights, 
            numpy_type=weights_type, 
            device_map=device, 
            quantization_config=bnb_config)
        outputs = []
        tokenizer = AutoTokenizer.from_pretrained(self.llm_weights)
        preprocess = MistralLoraTrainer.get_preprocess_callback(tokenizer, 128)
        llm.eval()
        self.logger.info("load llm weights")
        tk_dataset = dataset.map(preprocess)
        ds_loader = DataLoader(tk_dataset, batch_size=batch_s, shuffle=False, collate_fn=collate_fn)
        self.logger.info("starting generation...")
        for batch in ds_loader:
            input_ids= batch["input_ids"]
            attention_mask = batch["attention_mask"]
            generation = llm.generate(input_ids, attention_mask, max_length=100, pad_token_id=tokenizer.eos_token_id)
            outputs.extend(tokenizer.batch_decode(generation, skip_special_tokens=True))
        
        dataset = dataset.add_column("edited_prompt", outputs)
        
        if save_name:
            dataset.save_to_disk(self.base_path.joinpath(save_name))
            self.logger.info("save generations in:%s", str(self.base_path.joinpath(save_name)))
        
        self.new_dataset = dataset
        return outputs
    

    def generate_pair_images(self, prompts:list[str]|str|Path, device:str='cpu', save_name:str|None = None) -> list[tuple[PIL.Image, str, PIL.Image]]:
        pass

    def generate_new_dataset(self, num_samples:int, columns_prompt:str="original_prompt", save_mid_steps:bool=False, device:str='cpu', batch_s:int=8) -> DatasetDict:
        self.new_dataset = None
        edit_name = None 
        
        if save_mid_steps:
            edit_name = "edited_prompts"
        edited_prompts = self.generate_edit_prompts(num_samples, columns_prompt, device, batch_s, edit_name)

        pass


        
        

