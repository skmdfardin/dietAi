import json
from unsloth import FastLanguageModel
import torch

# Load the training data
with open('training_data.json', 'r') as f:
    training_data = json.load(f)

# Initialize the Ollama model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="ollama/llama2",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Prepare the dataset
train_data = [
    {
        "input_ids": tokenizer.encode(f"Input: {item['input']}\nOutput: {item['output']}", return_tensors="pt").squeeze(),
        "labels": tokenizer.encode(item['output'], return_tensors="pt").squeeze(),
    }
    for item in training_data
]

# Set up the training arguments
training_args = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 2,
    "max_steps": 10,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 1,
    "output_dir": "output",
    "optim": "adamw_8bit"
}

# Finetune the model
model.train()
model.finetune(train_data, **training_args)

# Save the finetuned model
model.save_pretrained("finetuned_nutrition_model")
