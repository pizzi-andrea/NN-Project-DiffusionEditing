# @title
from typing import List
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image
from attentionController import AttentionController, run_prompt_to_prompt
from ..libs.metrics.Clip import directional_similarity
import clip

class PromptToPromptGenerator:
    def __init__(self, stable_diffusion_model, num_inference_steps = 30, device = "cuda"):
        self.device = device
        self.num_inference_steps=num_inference_steps
        self.count = 0 

        self.pipe = stable_diffusion_model
        self.pipe = self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        
        self.controller = AttentionController(total_steps=self.num_inference_steps)
        self.CLIP_model, self.CLIP_preprocess = clip.load("ViT-B/32", device=device)

    def generate(self, prompt1, prompt2, p = 0.3, guidance_scale = 7.5, alpha = 0.7, path:str = None, CLIP = True):

        self.controller.setP(p)
        self.controller.setAlpha(alpha)

        img1, img2 = run_prompt_to_prompt(self.controller, 
                                          self.pipe, 
                                          prompt1, 
                                          prompt2,
                                          guidance_scale,
                                          self.num_inference_steps)

        if path is not None:
            img1.save(path + f"img_base_{self.count}.png")
            img2.save(path + f"img_edited_{self.count}.png")

        if CLIP:
            loss = directional_similarity(img1, img2,
                              prompt1, prompt2,
                              self.CLIP_model, 
                              self.CLIP_preprocess, 
                              self.device)
        else:
            loss = None
            
        self.count += 1

        return img1, img2, loss
        