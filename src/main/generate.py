import torch

from ..libs.DiTest import DiTest  # importa la classe/funzione dal modulo relativo
from argparse import ArgumentParser

# Models

# "stabilityai/stable-diffusion-3.5-large-turbo"
# stabilityai/sdxl-turbo

generator = DiTest("stabilityai/stable-diffusion-2-1", num_step=4)  # crea un'istanza della classe DiTest

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