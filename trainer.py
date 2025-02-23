import torch
import wandb
import argparse
import os
from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
import pandas as pd 
from unsloth import is_bfloat16_supported


class LlamaTrainer:
    def __init__(self, num_gpus=1):
        self.num_gpus = num_gpus
        self.dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.dataset = None
        self.trainer = None

    def initialize_wandb(self):
        """Initializes Weights & Biases logging."""
        if not wandb.api.api_key:
            print("Logging into Weights & Biases...")
            os.system("wandb login")  

        wandb.init(project="llama-training", name="llama_finetune", resume="allow")

    def load_model_and_tokenizer(self):
        """Loads the pre-trained model and tokenizer."""
        max_seq_length = 5020
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=self.dtype
        )

        # Apply LoRA and other modifications
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
            random_state=32,
            loftq_config=None,
        )

        print(self.model.print_trainable_parameters())

    def preprocess_data(self, dataset):
        """Applies chat template and tokenization to dataset."""

        def apply_chat_template(example):
            messages = [
                {"role": "user", "content": example['email']},
                {"role": "assistant", "content": example['response']}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return {"prompt": prompt}

        dataset = dataset.map(apply_chat_template)

        def tokenize_function(example):
            tokens = self.tokenizer(
                example['prompt'], padding="max_length", truncation=True, max_length=1024
            )
            tokens['labels'] = [
                -100 if token == self.tokenizer.pad_token_id else token for token in tokens['input_ids']
            ]
            return tokens

        tokenized_dataset = dataset.map(tokenize_function)
        tokenized_dataset = tokenized_dataset.remove_columns(['email', 'response', 'prompt'])
        return tokenized_dataset

    def get_training_arguments(self):
        """Returns training arguments for Hugging Face Trainer."""
        return TrainingArguments(
            output_dir="./results",
            evaluation_strategy="steps",
            eval_steps=100,
            logging_steps=5,
            save_steps=100,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=2,
            fp16=True if self.dtype == torch.float16 else False,  
            bf16=True if self.dtype == torch.bfloat16 else False,  
            report_to="wandb", 
            log_level="info",
            learning_rate=5e-5,
            max_grad_norm=2,
            save_total_limit=2,
            load_best_model_at_end=True,
            gradient_accumulation_steps=2, 
            dataloader_num_workers=4,
            warmup_steps=100,
        )

    class CustomTrainer(Trainer):
        """Trainer subclass that logs metrics to W&B."""
        def on_log(self, logs, **kwargs):
            super().on_log(logs, **kwargs)
            wandb.log(logs)

    def load_data(self):
        """Loads and preprocesses the dataset."""
        df = pd.read_csv('email_data_v1.csv')
        dataset = Dataset.from_pandas(df)
        dataset = dataset.train_test_split(0.05)
        self.dataset = self.preprocess_data(dataset)

    def setup_trainer(self):
        """Sets up the trainer with arguments and dataset."""
        self.training_args = self.get_training_arguments()
        self.trainer = self.CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            tokenizer=self.tokenizer,
        )

    def train(self):
        """Trains the model."""
        self.trainer.train()
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Distributed Training Script")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()

    trainer = LlamaTrainer(num_gpus=args.num_gpus)
    trainer.initialize_wandb()
    trainer.load_model_and_tokenizer()
    trainer.load_data()
    trainer.setup_trainer()
    trainer.train()


if __name__ == "__main__":
    main()
