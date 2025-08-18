from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
import torch
from pathlib import Path
from diffusers.utils.loading_utils import load_image
class DiTest:
    def __init__(self, model_id:str="stabilityai/stable-diffusion-2", num_step=4, guidance_scale:float = 0.5, strength:float=0) -> None:
        
        self.main = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            use_safetensors=True,
            device_map='balanced'
        )

        self.main.scheduler = DPMSolverMultistepScheduler.from_config(self.main.scheduler.config)

        #self.main.enable_model_cpu_offload()
        self.default_num_step = num_step
        self.default_guidance = guidance_scale
        self.strength = strength
        
        if getattr(self.main, "unet", None):
            self.main.unet = torch.compile(self.main.unet, mode="reduce-overhead", fullgraph=True)

    def generate(self, prompt:str, reference_path:str|Path|None=None):
        
        if reference_path:
            reference_path = Path(reference_path)
            if not reference_path.exists:
                raise FileNotFoundError
            
            ref_image = load_image(str(reference_path))
        else:
            ref_image = None
        
        print(f"Number steps for generation: {self.default_num_step}")
        print(f"stregth value: {self.strength}")
        print(f"Guidance scale value: {self.default_guidance}")
        gen = self.main(
            prompt=prompt,
            num_inference_steps=self.default_num_step,
            guidance_scale=self.default_guidance,
            strength=self.strength,
            max_sequence_length=512,
            image=ref_image
        ).images[0]
        gen.save(f"{prompt.__hash__()}.png")
        return gen
