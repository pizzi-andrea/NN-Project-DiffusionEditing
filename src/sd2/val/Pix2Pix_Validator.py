from pathlib import Path
import PIL.Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from os import mkdir, makedirs
from tqdm.auto import tqdm
import torch
import PIL
class Pix2Pix_Validator:
    def __init__(self,  pipline_weights:Path|str, out_dir:Path|str, device:str='cpu', dtype=torch.bfloat16) -> None:
        
        self.pipline =  StableDiffusionInstructPix2PixPipeline.from_pretrained(
            pipline_weights,
            torch_dtype=dtype,
            map_device=device
        )
       
        self.pipline.enable_sequential_cpu_offload()
        self.out_dir = Path(out_dir)
        
        if not self.out_dir.exists():
            mkdir(out_dir)
    
    def edit(self, path_to_image:str|Path, prompt:str, num_generations:int, reference_img:str|Path|None=None,prompt_guidance:float=1.0, image_guidance:float=1.0) -> list[PIL.Image.Image]:
        guidance_img = PIL.Image.open(path_to_image).convert("RGB")
        target_img = None
        generations = []

        for idx in tqdm(range(0, num_generations), desc="Editing image"):
            
            img = self.pipline(
                prompt=prompt,
                image=guidance_img,
                num_inference_steps = 40,
                prompt_guidance=prompt_guidance,
                image_guidance_scale=image_guidance,
                streght=0.8
                
            ).images[0]

            img.save(self.out_dir.joinpath(f"image_generated_{idx}.png"))
            generations.append(img)
        
        return generations
        

        