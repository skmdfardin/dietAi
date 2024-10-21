import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('nutrition.csv')

# Replace null values with 0
df = df.fillna(0)

# Additional cleaning steps can be added here
# For example:
# - Remove duplicates
df = df.drop_duplicates()

# - Convert data types if needed
# df['column_name'] = df['column_name'].astype(int)

# Save the cleaned dataset
df.to_csv('cleaned_dataset.csv', index=False)

print("Data cleaning completed. Cleaned dataset saved as 'cleaned_dataset.csv'")
