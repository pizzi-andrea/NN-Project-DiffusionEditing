from typing import Callable, Literal
import torch 
import open_clip as clip

class ClipWrap:

    """
    Wrapper class for OpenClip methods used to improve multi-modal rappresentation
    """
    @staticmethod
    def info():
        print(clip.list_pretrained())
        
    def __init__(self,photo_encoder:str, tokenizer:str, precision:str="float32", weights:str|None=None, device:Literal['cpu', 'cuda']="cpu"):
        self.precision = precision
        self.device = device
        
        self.encoder, train_arg, val_arg = clip.create_model_and_transforms(
            model_name=photo_encoder,
            pretrained=weights,
            load_weights=True,
            device=self.device,
            precision=self.precision,
            jit=True
        )

        self.train_arg = train_arg
        self.val_arg = val_arg

        self.tokenizer = clip.get_tokenizer(tokenizer)
        #print(self.encoder)
    
    def get_transformations(self) -> tuple[Callable, Callable]:
        return (self.train_arg, self.val_arg) # type: ignore
    
    def compute(self, images, prompts):
        self.encoder.eval()
        
        with torch.no_grad():
            
            # Preprocessing prompts
            tokens = self.tokenizer(prompts).to(self.device)
            prompts_emb = self.encoder.encode_text(tokens) # type: ignore
            prompts = prompts_emb.to(getattr(torch, self.precision)).to(self.device)
            
            # Preprocessing images
            #images = self.val_arg(images)
            images  = images.to(getattr(torch, self.precision)).to(self.device)
            images_emb = self.encoder.encode_image(images) # type: ignore
            
            
            img_std, img_mean = torch.std_mean(images_emb, dim=-1, keepdim=True)
            p_std, p_mean = torch.std_mean(prompts_emb, dim=-1, keepdim=True)

            #print(f"img_emb:{images.shape}")
            #print(f"prompt_emb:{prompts.shape}")

            images = (images_emb - img_mean)/img_std
            prompts = (prompts_emb - p_mean)/p_std

            coss = torch.cosine_similarity(prompts, images, dim=-1)

        return [images_emb, prompts_emb, coss]