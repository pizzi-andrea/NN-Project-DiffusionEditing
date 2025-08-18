import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk, DatasetDict



class MistralLoraTrainer:
    """
    Wrapper class for Mistral AI models traning with Lora Quantization method
    """
    def __init__(self, model_name, dataset_path, device, max_length=128, enabled_split=["train","test","validation"], aggresive_quantization=False, metrics_callback=None):
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.metrics_callback = metrics_callback
        self.split = enabled_split
        
        if aggresive_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,  # or torch.float16 if bfloat16 not supported
            )
            map_device = 'auto'
        else:
            bnb_config = None
            map_device = None
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._load_dataset(dataset_path, 0.01)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=map_device, quantization_config=bnb_config)

        self.preprocess = self.get_preprocess_callback(self.tokenizer, max_length=self.max_length)
    
    @staticmethod
    def get_preprocess_callback(tokenizer, max_length):
        def preprocess(example):
            prompt = example["original_prompt"] + "\n##\n"
            completion = example["edit_prompt"] + "\n%%\n" + example["edited_prompt"] + "\nEND"
            inputs = tokenizer(
                prompt + completion,
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        return preprocess

    def _load_dataset(self, dataset_path, limit=1.0):

        if limit > 1 or limit <= 0:
            raise ValueError("Limitation factor bust be between 0 and 1")
        
        
        
        ds = {split:load_from_disk(dataset_path+'/'+split) for split in self.split}
        self.dataset = DatasetDict(ds)

        if limit < 1:
            print(f"load %{limit*100} of train dataset: {int(len(self.dataset['train'])*limit)} items")
            print(f"load %{limit*100} of validation dataset: {int(len(self.dataset['validation'])*limit)} items")

        self.train_dataset = self.dataset['train'].select(range( int(len(self.dataset['train'])*limit)))
        self.validation_dataset = self.dataset['validation'].select(range(int(len(self.dataset['validation'])*limit)))
                

    def apply_lora(self, r=8, alpha=16, target_modules=["q_proj", "v_proj"], dropout=0.1):
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            inference_mode=False,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

    def train(self, output_dir="./mistral-lora-output", epochs=3, batch_size=8, lr=2e-4):
        self.tokenized_train_dataset = self.train_dataset.map(self.preprocess,
                                                              remove_columns=["original_prompt", "edit_prompt", "edited_prompt"])
        
        self.tokenized_validation_dataset = self.validation_dataset.map(self.preprocess,
                                                              remove_columns=["original_prompt", "edit_prompt", "edited_prompt"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            learning_rate=lr,
            #logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=True,
            remove_unused_columns=False,
            report_to="none",
            eval_accumulation_steps=1,  # force metrics computation on cpu
            load_best_model_at_end=True,
            save_total_limit=3,
            metric_for_best_model="rouge1"
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train_dataset, # Use tokenized dataset
            eval_dataset=self.tokenized_validation_dataset,
            processing_class=self.tokenizer, # Use processing_class instead of tokenizer
            data_collator=data_collator,
            compute_metrics=self.metrics_callback,
            
            
        )

        trainer.train()
        # load best model
        self.model = trainer.model
        self.model.save_pretrained(output_dir+'/'+'best_weigths')

        return self.model

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_length=200, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
