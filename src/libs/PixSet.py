from typing import *
from torch.utils.data import Dataset
from datasets import load_dataset
from pathlib import Path
import numpy as np
import cv2 as cv
class PixSet(Dataset):

    def __init__(self, src_dataset:Path|str,size:int, split:Literal['train','val','test'],offset:int=0, transformation:Callable|None = None):
        super().__init__()
        src_dataset = Path(src_dataset)
        self.offset = offset
        self.transformation = transformation
        if not src_dataset.exists():
            raise FileNotFoundError
        
        self.ds = load_dataset(path=str(src_dataset), split=f'{split}[{offset}:{size}]')
        
        if offset + size > self.ds.dataset_size:
            raise IndexError
        
    def __len__(self):
        return len(self.ds)
    
    def get_hf_dataset(self):
        return self.ds
    

    def __getitem__(self, index):
        record = self.ds[self.offset + index]
        
     
        original = record['original_image']
        edited   = record['edited_image']

        if self.transformation:
            original = self.transformation(original)
            edited   = self.transformation(edited)
        
        sample = {
            "original_image" : original,
            "original_prompt" : record['original_prompt'],
            "edit" : record['edit_prompt'],
            "edited_prompt": record['edited_prompt'],
            "edited_image" : edited
        }

        return sample