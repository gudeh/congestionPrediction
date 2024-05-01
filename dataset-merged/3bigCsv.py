import os
import pandas as pd

def merge_csv_files(root_directory, output_file, csv_filename):
    # DataFrame list to hold all dataframes before concatenation
    df_list = []
    # Define the columns to drop
    columns_to_drop = ['xMin', 'yMin', 'xMax', 'yMax', 'conCount', 'placementHeat', 'irDropHeat', 'powerHeat', 'xMin_x','yMin_x','xMax_x','yMax_x','xMin_y','yMin_y','xMax_y','yMax_y']
    # Walk through the directory
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file == csv_filename:
                # Load the CSV file
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                # Remove rows where 'Eigen' is NaN
                df.dropna(subset=['Eigen'], inplace=True)
                # Drop unwanted columns if they exist in DataFrame
                df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')
                # Extract design name from the path
                design_name = os.path.basename(subdir)
                # Extract dataset name from the path and set as 'A' or 'B'
                dataset_name = os.path.basename(os.path.dirname(subdir))
                dataset_label = 'A' if dataset_name == 'nangateV2' else 'B'
                # Add the new columns
                df['Dataset'] = dataset_label
                df['Design'] = design_name
                # Append the dataframe to the list
                df_list.append(df)
    
    # Concatenate all dataframes into one
    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        # Save to a new CSV file
        full_df.to_csv(output_file, index=False)
        print(f"Created consolidated file: {output_file}")
    else:
        print(f"No files found for {csv_filename}")

# Paths to the dataset directories
dataset_a_path = './nangateV2'
dataset_b_path = './asap7V2'

# Call the function for each dataset and CSV type
merge_csv_files(dataset_a_path, 'A_gatesToHeatSTDfeatures.csv', 'gatesToHeatSTDfeatures.csv')
merge_csv_files(dataset_a_path, 'A_preProcessedGatesToHeat.csv', 'preProcessedGatesToHeat.csv')
merge_csv_files(dataset_b_path, 'B_gatesToHeatSTDfeatures.csv', 'gatesToHeatSTDfeatures.csv')
merge_csv_files(dataset_b_path, 'B_preProcessedGatesToHeat.csv', 'preProcessedGatesToHeat.csv')
