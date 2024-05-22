import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_correlation_matrix_plot(csv_file_path, output_correlation_image):
    # Load data from CSV
    df = pd.read_csv(csv_file_path)

    # Filter out rows where the 'type' column contains 'fakeram'
    df = df[~df['type'].str.contains('fakeram', na=False)]

    # Columns to exclude
    columns_to_exclude = ['name', 'type', 'logic_function', 'Dataset', 'Design']
    df.drop(columns=columns_to_exclude, inplace=True, errors='ignore')

    # Convert all columns that should be numeric but are read as objects
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any remaining NaNs
    df.dropna(inplace=True)

    # Correlation matrix
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numerical_cols].corr()

    # Save correlation matrix plot to file
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'shrink': .75})
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(output_correlation_image)
    plt.close()  # Close the plot to free memory

# Example usage
create_correlation_matrix_plot('merged.csv', 'correlation_heatmap.png')
