import os
import pandas as pd
import glob

# Define the directory path where CSV files are stored
path = 'DATA/ids-small-part/'  # Replace with your path
files = glob.glob(os.path.join(path, "test.csv*"))

# Loop through the files and process each one
for i, file in enumerate(files):
    # Read the CSV file
    df = pd.read_csv(file)

    # Drop the 'Label' column
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    
    # Save the dataframe to a new CSV file with the updated name
    new_file_name = f'cse_cic_ids2018_part_{i}.csv'
    df.to_csv("app/data_csv/"+new_file_name, index=False)

    # Optional: Print out the new file path to confirm it's saved
    print(f"File saved: {new_file_name}")
