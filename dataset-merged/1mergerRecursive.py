import os
import pandas as pd

def merge_csv_files(directory):
    # Format the filename with the directory name
    directory_name = os.path.basename(directory)
    position_file = f'{directory}/gatesPosition_{directory_name}.csv'

    if os.path.exists(position_file):
        # Load the position CSV file
        df_position = pd.read_csv(position_file)
        # Standardize column names in the position file
        df_position.rename(columns={'Name': 'name'}, inplace=True)

        # List of other CSV files to merge with
        files_to_merge = ['gatesToHeatSTDfeatures.csv', 'preProcessedGatesToHeat.csv']
        files_to_merge = [f'{directory}/{file}' for file in files_to_merge if os.path.exists(f'{directory}/{file}')]

        # Loop through the files and merge them
        for file in files_to_merge:
            df_other = pd.read_csv(file)
            merged_df = pd.merge(df_position, df_other, on='name', how='inner')
            # Overwrite the original file with the merged data
            merged_df.to_csv(file, index=False)

def traverse_and_merge(root_directory):
    # Walk through all subdirectories
    for subdir, dirs, files in os.walk(root_directory):
        merge_csv_files(subdir)

# Specify the root directory
root_directory = './'
traverse_and_merge(root_directory)
