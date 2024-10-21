"""
Finetunes the Ollama language model using the provided training data.

The `finetune_model` function sends a request to the Ollama API to finetune the model using the provided prompt and completion. The function logs the input and output of the finetuning process.

The script loads the training data from a JSON file and iterates through each example, calling the `finetune_model` function to finetune the model. Finally, it logs a message indicating that the finetuning process is complete.
"""
import json
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the training data
with open('ollama_training_data.json', 'r') as f:
    training_data = json.load(f)

# Ollama API endpoint
OLLAMA_API_URL = 'http://localhost:11434/api/generate'

# Function to finetune the model
import time

def finetune_model(prompt, completion, max_retries=3):
    data = {
        "model": "llama3.2",
        "prompt": f"Input: {prompt}\nOutput: {completion}",
        "system": "You are being finetuned to provide nutritional information."
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(OLLAMA_API_URL, json=data)
            response.raise_for_status()
            result = response.json()
            if result['message']['content']:
                return result
            else:
                logging.warning(f"Empty response on attempt {attempt + 1}. Retrying...")
                time.sleep(1)  # Wait for 1 second before retrying
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1)
    
    logging.error("Max retries reached. Unable to get a valid response.")
    return None
# Finetune the model with each example in the training data
for example in training_data:
    input_val = example['input']
    output_val = example['output']
    
    logging.info(f"Input: {input_val}")
    result = finetune_model(input_val, output_val)
    if result:
        logging.info(f"Output: {result.get('message', {}).get('content', '')}")
    else:
        logging.info("Failed to get a valid response from the API")
    logging.info("---")

logging.info("Finetuning completed!")
