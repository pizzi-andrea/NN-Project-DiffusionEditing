from datasets import load_from_disk, DatasetDict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
from ..libs.mistralLoraTrainer import MistralLoraTrainer

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = MistralLoraTrainer(
        model_name="mistralai/Mistral-7B-v0.1",
        dataset_path="L_spixset/train",
        device=DEVICE,
        max_length=128
    )

    trainer.apply_lora(r=8, alpha=16, target_modules=["q_proj", "v_proj"], dropout=0.1)


    trainer.model.print_trainable_parameters()

    trainer.train()

    validation_dataset = trainer.validation_dataset

    for i in range(10):
      print(f"--------- PRINTING SENTENCE {i+1} ---------")
      prompt = validation_dataset[i]['original_prompt'] + "\n##\n"
      print(f"initial sentence -> {prompt}")

      print(f"EDIT \n")

      out = trainer.generate(prompt)

      edit_prompt = out.split("\n##\n")[1].split("\n%%\n")[0]
      edited_prompt = out.split("\n%%\n")[1].split("\nEND")[0]

      print(f"edit prompt -> {edit_prompt}")
      print(f"GT edit prompt -> {validation_dataset[i]['edit_prompt']}\n")

      print(f"edited prompt -> {edited_prompt}")
      print(f"GT edited prompt -> {validation_dataset[i]['edited_prompt']}\n")
