import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_csv_files(directory):
    logging.info(f"Processing directory: {directory}")
    directory_name = os.path.basename(directory)
    
    # Identify main files to merge with
    main_files = ['preProcessedGatesToHeat.csv', 'gatesToHeatSTDfeatures.csv']
    main_files = [f'{directory}/{file}' for file in main_files if os.path.exists(f'{directory}/{file}')]

    # Identify the smaller files that start with the directory name
    small_files = [f for f in os.listdir(directory) if f.startswith(directory_name) and f.endswith('.csv')]
    small_files = [f'{directory}/{file}' for file in small_files]

    # Process each main file
    for main_file in main_files:
        df_main = pd.read_csv(main_file)
        # Merge with each relevant small file
        for small_file in small_files:
            df_small = pd.read_csv(small_file)
            # Find the ID column (starts with 'id')
            id_column = next(col for col in df_small.columns if col.startswith('id'))

            # Version 1:
            # # Filter df_small to include only 'feat' columns
            # feat_columns = [col for col in df_small.columns if col.startswith('feat')]
            # # Rename 'feat' columns to keep only the last part after the last underscore
            # df_small.rename(columns={col: col.split('-')[-1] for col in feat_columns}, inplace=True)
            # # Update the feat_columns with new names
            # feat_columns = [col.split('-')[-1] for col in feat_columns]
            # # Select ID and feature columns for merging
            # df_small_reduced = df_small[[id_column] + feat_columns]

            # Version for ID, IO, P, E:
            # Specific columns to merge
            columns_to_merge = ['Eigen', 'pageRank', 'inDegree', 'outDegree']
            # Ensure only specified columns (if they exist) and the ID are included for merging
            columns_to_include = [id_column] + [col for col in columns_to_merge if col in df_small.columns]
            df_small_reduced = df_small[columns_to_include]
            
            # Merge on the identified 'id' column, using outer join
            df_main = pd.merge(df_main, df_small_reduced, left_on='id', right_on=id_column, how='outer')
            # Drop the redundant 'id' column from df_small
            df_main.drop(columns=[id_column], inplace=True)
        
        # Overwrite the original main file with the merged data
        df_main.to_csv(main_file, index=False)
        logging.info(f"Merged and overwritten: {main_file}")

def traverse_and_merge(root_directory):
    for subdir, dirs, files in os.walk(root_directory):
        merge_csv_files(subdir)

if __name__ == "__main__":
    root_directory = './'
    traverse_and_merge(root_directory)
