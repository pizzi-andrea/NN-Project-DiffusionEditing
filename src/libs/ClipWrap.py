from typing import Literal
import torch 
import cv2 as cv
import numpy as np
import open_clip as clip

class ClipWrap:
    def __init__(self,photo_encoder:str, tokenizer:str, precision:str="f32", weights:str|None=None, device:Literal['cpu', 'cuda']="cpu"):
        self.precision = precision
        self.device = device


        
        self.encoder = clip.create_model(
            model_name=photo_encoder,
            pretrained=weights,
            device=self.device,
            precision=self.precision
        )

        self.tokenizer = clip.get_tokenizer(tokenizer, context_length=512)
    
   

    def train(self, dataloader:torch.utils.data.DataLoader) -> dict[str,list[int]]:
        pass

    def eval(self, dataloader:torch.utils.data.DataLoader) -> dict[str,list[int]]:
        pass 

    

    def compute(self, images, prompts):
        self.model.eval()

        with torch.no_grad():
            images = self.model.encode_image(images)
            prompts = self.model.encode_text(self.tokenizer.tokenizer.tokenize(prompts))
            img_std, img_mean = torch.std_mean(images, dim=-1, keepdim=True)
            p_std, p_mean = torch.std_mean(prompts, dim=-1, keepdim=True)

            images = (images - img_mean)/img_std
            prompts = (prompts - p_mean)/p_std

            embeddings = (images @ prompts.T)

        return embeddings
    








