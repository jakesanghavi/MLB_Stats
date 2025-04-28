import pandas as pd
import glob

# Set the year you are interested in
year=2025

# Get a list of all CSV files in the folder
folder_path = f'Games/{year}'
all_files = glob.glob(folder_path + "/*.csv")

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through the list of file names and read each CSV into a DataFrame
for filename in all_files:
    dfs.append(pd.read_csv(filename))

# Concatenate all DataFrames into one
combined_df = pd.concat(dfs, ignore_index=True)

# Write the new DF to an appropriate location
combined_df.to_csv(f'{year}_full_pbp.csv', index=False)
