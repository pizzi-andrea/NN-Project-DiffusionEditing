import torch

from ..libs.DiTest import DiTest  # importa la classe/funzione dal modulo relativo
from argparse import ArgumentParser
from diffusers.quantizers.quantization_config import BitsAndBytesConfig

# Models

# "stabilityai/stable-diffusion-3.5-large-turbo"
# stabilityai/sdxl-turbo

generator = DiTest("stabilityai/sdxl-turbo", guidance_scale=8, num_step=25, strength=0.9)  # crea un'istanza della classe DiTest

parser = ArgumentParser(description="input")
if __name__ == "__main__":
    parser.add_argument(
        '-p', '--prompt', help="prompt guidline", required=True, type=str
    )

    parser.add_argument(
        '-r', '--reference', help="Reference image", required=False, type=str
    )

    args = parser.parse_args()
    m = args.prompt
    ref = args.reference

    generator.generate(m, ref)