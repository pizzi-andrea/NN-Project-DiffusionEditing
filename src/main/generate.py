

from ..libs.DiTest import DiTest  # importa la classe/funzione dal modulo relativo
from argparse import ArgumentParser

generator = DiTest("stabilityai/sdxl-turbo")  # crea un'istanza della classe DiTest

parser = ArgumentParser(description="input")
if __name__ == "__main__":
    parser.add_argument(
        '-p', '--prompt', help="prompt guidline", required=True, type=str
    )

    args = parser.parse_args()
    m = args.prompt

    generator.generate(m)