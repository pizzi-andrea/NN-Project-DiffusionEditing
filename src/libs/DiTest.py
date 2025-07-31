from diffusers import BitsAndBytesConfig, AutoPipelineForText2Image
from diffusers import StableDiffusion3Pipeline
import torch

class DiTest:
    def __init__(self, model_id:str="stabilityai/stable-diffusion-3.5-large") -> None:
        self.conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
        

        self.main = AutoPipelineForText2Image.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        )
        self.main.enable_model_cpu_offload()
    
    def generate(self, prompt:str):
        image = self.main(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=5.0,
            max_sequence_length=512,
        ).images[0]
        image.save(f"{prompt.__hash__()}.png")
        return image
