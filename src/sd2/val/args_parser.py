import argparse
from pathlib import Path
def args_parse():
    parser = argparse.ArgumentParser(description="Parameters for Eval Instruct_Pix2Pix model")

    parser.add_argument(
        '--photo', help='path to single photo', required=True, type=str
    )

    parser.add_argument(
        '--ref_photo', help='path to reference photo use to compute loss value', required=False, type=str
    )

    parser.add_argument(
        '--num_edits', help='number of edits performe during generation', required=False, default=3, type=int
    )

    parser.add_argument(
        '--edit_prompt', help='edit prompt', required=True, type=str
    )

    parser.add_argument(
        '--pipline_weights', help='path to InstructPix2Pix pipline weights', required=True, type=str
    )

    parser.add_argument(
        '--out_dir', help='directory where save validation data', required=True, type=str
    )

    parser.add_argument(
        '--guidance_prompt_scale', help='prompt guidance', required=False, default=1, type=float
    )

    parser.add_argument(
        '--guidance_image_scale', help='image guidance', required=False, default=1, type=float
    )

    args = parser.parse_args()

    # sanity check

    # ...

    return args
