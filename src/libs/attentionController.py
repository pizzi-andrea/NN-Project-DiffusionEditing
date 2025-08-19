import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline

class AttentionController:
    def __init__(self, total_steps:int, p:float=0.7, alpha:float=0.7):
        self.total_steps = total_steps
        self.setP(p)
        self.alpha = alpha
        self.current_step = 0
        self.phase = "save"  # oppure "replace"
        self.saved_self_attn = []
        self.saved_cross_attn = []
        self.hooks = []

    # ---------- hook fn ----------
    def save_self_attention(self, module, inputs, output):
        if self.current_step < self.shared_steps:
            self.saved_self_attn.append(output.detach().clone())
        return output

    def replace_self_attention(self, module, inputs, output):
        if self.current_step < self.shared_steps and self.saved_self_attn:
            saved = self.saved_self_attn.pop(0).to(output.device)
            out = self.alpha * saved + (1 - self.alpha) * output
            del saved
            return out
        return output

    def save_cross_attention(self, module, inputs, output):
        if self.current_step >= self.shared_steps:
            self.saved_cross_attn.append(output.detach().clone())
        return output

    def replace_cross_attention(self, module, inputs, output):
        if self.current_step >= self.shared_steps and self.saved_cross_attn:
            saved = self.saved_cross_attn.pop(0).to(output.device)
            out = saved
            del saved
            return output #out
        return output

    def setAlpha(self, value:float):
        assert 0.0 <= value <= 1.0
        self.alpha = value

    def setP(self, value:float):
        assert 0.0 <= value <= 1.0
        self.p = value
        self.shared_steps = int(round(self.p * self.total_steps))

    def set_phase(self, phase:str):
        assert phase in ("save", "replace")
        self.phase = phase

    def reset_step_counter(self):
        self.current_step = 0

    def on_step(self, step_idx:int):
        # chiamato dal callback della pipeline
        self.current_step = step_idx

    def register_hooks(self, pipe, save:bool):
        # self-attn = attn1, cross-attn = attn2 (UNet2DConditionModel in diffusers)
        for name, module in pipe.unet.named_modules():
            if "attn1" in name:
                self.hooks.append(
                    module.register_forward_hook(
                        self.save_self_attention if save else self.replace_self_attention
                    )
                )
            elif "attn2" in name:
                self.hooks.append(
                    module.register_forward_hook(
                        self.save_cross_attention if save else self.replace_cross_attention
                    )
                )

    def remove_hooks_and_clear(self, clear_buffers:bool=False):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        if clear_buffers:
            self.saved_self_attn.clear()
            self.saved_cross_attn.clear()

def run_prompt_to_prompt(
    controller: AttentionController,
    pipe: StableDiffusionPipeline,
    prompt1: str,
    prompt2: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
):
    assert num_inference_steps == controller.total_steps, (
        "num_inference_steps deve combaciare con total_steps del controller"
    )

    # ---- Pass 1: SAVE ----
    controller.set_phase("save")
    controller.reset_step_counter()
    controller.remove_hooks_and_clear(clear_buffers=True)
    controller.register_hooks(pipe, save=True)

    def cb_save(pipe, step_index, timestep, callback_kwargs):
        controller.on_step(step_index)
        return callback_kwargs

    with torch.inference_mode():
        img1 = pipe(
            prompt1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            added_cond_kwargs={},
            callback_on_step_end=cb_save,
        ).images[0]

    controller.remove_hooks_and_clear(clear_buffers=False)  # conserva i buffer per il replace

    # ---- Pass 2: REPLACE ----
    controller.set_phase("replace")
    controller.reset_step_counter()
    controller.register_hooks(pipe, save=False)

    def cb_replace(pipe, step_index, timestep, callback_kwargs):
        controller.on_step(step_index)
        return callback_kwargs

    with torch.inference_mode():
        img2 = pipe(
            prompt2,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            added_cond_kwargs={},
            callback_on_step_end=cb_replace,
        ).images[0]

    controller.remove_hooks_and_clear(clear_buffers=True)

    return img1, img2
