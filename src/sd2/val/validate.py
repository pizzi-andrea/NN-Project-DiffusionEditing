from ast import arg
from args_parser import args_parse
from Pix2Pix_Validator import Pix2Pix_Validator
import torch
if __name__ == "__main__":
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    args = args_parse()

    validator = Pix2Pix_Validator(args.pipline_weights, args.out_dir, device)
    validator.edit(args.photo, args.edit_prompt, args.num_edits, prompt_guidance=args.guidance_prompt_scale, image_guidance=args.guidance_image_scale)
