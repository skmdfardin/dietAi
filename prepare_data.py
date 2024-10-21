import pandas as pd
import json

# Load the cleaned CSV file
df = pd.read_csv('cleaned_dataset.csv')

# Function to create the output string
def create_output(row):
    output = "100gms of {} contains as follows:\n".format(row['Food'])
    for column in df.columns:
        if column != 'Food' and column != 'Category' and column != 'Type':
            output += "{}: {} {}\n".format(column, row[column], 'g' if column != 'Energy' else 'kcal')
    return output.strip()

# Create a list to store our formatted data
formatted_data = []

# Iterate through each row in the dataframe
for index, row in df.iterrows():
    food_name = row['Food'].lower()
    category = row['Category'].lower() if 'Category' in row else ''
    food_type = row['Type'].lower() if 'Type' in row else ''
    
    # Create the input string
    input_str = f"{food_name}"
    if category:
        input_str = f"{category} {input_str}"
    if food_type:
        input_str = f"{input_str} {food_type}"
    
    # Create the output string
    output_str = create_output(row)
    
    # Add to our formatted data
    formatted_data.append({
        "input": input_str,
        "output": output_str
    })

# Save the formatted data as JSON
with open('nutrition_data.json', 'w') as f:
    json.dump(formatted_data, f, indent=2)

print("Data preparation completed. Dataset saved as 'nutrition_data.json'")
