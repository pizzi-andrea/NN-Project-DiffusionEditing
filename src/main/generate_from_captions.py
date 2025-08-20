import torch
from ..libs.promptToPrompt import PromptToPromptGenerator
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import AutoPipelineForText2Image
import random

if __name__ == "__main__":
    MODEL_ID = "stabilityai/stable-diffusion-2-1"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_INFERENCE_STEPS = 30
    ALPHA = 0.7
    GUIDANCE_SCALE = 7.5
    
    NUM_GEN = 20

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    #pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    ptp = PromptToPromptGenerator(pipe, num_inference_steps=NUM_INFERENCE_STEPS, device=DEVICE)

    max_score = 1

    prompt1 = "a girl riding a horse"
    prompt2 = "a girl riding a cyborg-horse"

    for i in range(NUM_GEN):
        print(f"--------- PRINTING SENTENCE {i+1} ---------")
        torch.cuda.empty_cache()
        p = random.uniform(0.1, 0.9)
        print(f"p = {p}")

        img1, img2, score = ptp.generate(prompt1,
                    prompt2, 
                    p = p,
                    alpha = ALPHA,
                    guidance_scale = GUIDANCE_SCALE,
                    #path = "./"
                    )
        
        print(f"score = {score}")

        plt.figure(figsize=(10,5))

        plt.subplot(1, 2, 1)
        plt.title(prompt1)
        plt.axis('off')
        plt.imshow(img1)

        plt.subplot(1, 2, 2)
        plt.title(prompt2)
        plt.axis('off')
        plt.imshow(img2)

        plt.show()

        #print(score)

        if score < max_score:
            print(f"new max score: {score}")
            max_score = score
            bst_img1 = img1
            bst_img2 = img2