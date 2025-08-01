from datasets import Dataset
from ..libs.PixSet import PixSet

if __name__ == "__main__":
    dataset = PixSet("dataset/pixset", 15_000, "train", offset=0, transformation=None)

    hf_dataset:Dataset = dataset.get_hf_dataset()
    hf_dataset.shuffle(seed=2025)

    # logging data
    print(f"dataset size -> {hf_dataset}")
    
    train = hf_dataset.select(range(0,10_000))
    validation = hf_dataset.select(range(10_000,12_500))
    test = hf_dataset.select(range(12_500,15_000)).select_columns(["original_prompt", "original_image"])

    train.save_to_disk("dataset/spixset/train", num_shards=500, num_proc=12)
    validation.save_to_disk("dataset/spixset/validation", num_shards=500, num_proc=12)
    test.save_to_disk("dataset/spixset/test", num_shards=500, num_proc=12)

    train.select_columns(["original_prompt", "edit_prompt", "edited_prompt"]).save_to_disk("dataset/L_spixset/train", num_shards=500, num_proc=12)
    validation.select_columns(["original_prompt", "edit_prompt", "edited_prompt"]).save_to_disk("dataset/L_spixset/validation", num_shards=500, num_proc=12)
    test.select_columns(["original_prompt"]).save_to_disk("dataset/L_spixset/test", num_shards=500, num_proc=12)


    

    
    
    