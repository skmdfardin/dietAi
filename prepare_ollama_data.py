import pandas as pd
import json
import requests

# Load the cleaned CSV file
df = pd.read_csv('cleaned_dataset.csv', index_col=0)

# Function to process each row
def process_row(row):
    name = row['name']
    nutrient_data = []
    for column, value in row.items():
        if column != 'name':
            nutrient_data.append([value, column])
    
    input_text = f"Find nutritional information for {name}"
    output_text = f"100gms of {name} contains the following nutrients:\n"
    output_text += "\n".join([f"{value} {nutrient}" for value, nutrient in nutrient_data])
    
    return {
        "input": input_text,
        "output": output_text
    }

processed_data = [process_row(row) for _, row in df.iterrows()]

# Save processed data to JSON file
with open('ollama_training_data.json', 'w') as f:
    json.dump(processed_data, f, indent=2)

print("Data preparation completed. Training data saved as 'ollama_training_data.json'")

# Function to send data to Ollama API and log responses
def send_to_ollama_and_log(data):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    responses = []
    
    for item in data:
        payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": item["input"]},
                {"role": "assistant", "content": item["output"]}
            ]
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print(f"Successfully sent: {item['input']}")
            responses.append({
                "input": item["input"],
                "expected_output": item["output"],
                "model_response": response.json()["message"]["content"]
            })
        else:
            print(f"Failed to send: {item['input']}")
    
    # Log responses
    with open('ollama_responses.json', 'w') as f:
        json.dump(responses, f, indent=2)

# Send data to Ollama API and log responses
send_to_ollama_and_log(processed_data)

print("Data sent to Ollama API and responses logged in 'ollama_responses.json'")

# Prepare data for Ollama fine-tuning
def prepare_ollama_finetune_data(data):
    finetune_data = []
    for item in data:
        finetune_data.append(f"Human: {item['input']}\n\nAssistant: {item['output']}\n\nHuman: Thank you\n\nAssistant: You're welcome! Is there anything else you'd like to know about nutrition?")
    
    with open('ollama_finetune.txt', 'w') as f:
        f.write('\n\n'.join(finetune_data))

prepare_ollama_finetune_data(processed_data)

print("Fine-tuning data prepared and saved as 'ollama_finetune.txt'")

# Instructions for fine-tuning
print("\nTo fine-tune the Ollama llama3.2 model:")
print("1. Ensure Ollama is installed and running")
print("2. Open a terminal and run the following command:")
print("   ollama create nutritionai -f ./Modelfile")
print("3. This will create a new model named 'nutritionai' based on llama3.2")
print("4. The fine-tuning process will start automatically")
print("5. Once completed, you can use the model with: ollama run nutritionai")

# Create Modelfile
modelfile_content = """
FROM llama3.2

# Set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 1

# Set the system prompt
SYSTEM You are an AI assistant specialized in providing nutritional information about various foods.

# Fine-tuning data
TRAIN ollama_finetune.txt
"""

with open('Modelfile', 'w') as f:
    f.write(modelfile_content)

print("\nModelfile created. You can now proceed with the fine-tuning process.")