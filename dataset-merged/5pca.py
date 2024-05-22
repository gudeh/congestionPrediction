import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_pca_plots(csv_file_path, output_pca_image_base):
    # Load data from CSV
    df = pd.read_csv(csv_file_path)

    # Filter out rows where the 'type' column contains 'fakeram'
    df = df[~df['type'].str.contains('fakeram', na=False)]

    # Columns to exclude from PCA but retain for other uses
    columns_to_exclude = ['name', 'type', 'logic_function', 'Dataset', 'Design']
    df.drop(columns=columns_to_exclude, inplace=True, errors='ignore')

    # Convert all columns that should be numeric but are read as objects
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any remaining NaNs and reset index
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)  # Resetting index after dropping NaNs

    # Standardizing the features for PCA
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features_scaled = StandardScaler().fit_transform(df[numerical_cols])

    # PCA
    pca = PCA(n_components=2)  # Reduce to 2 components
    principalComponents = pca.fit_transform(features_scaled)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    # Print the percentage of variance explained by each component
    print("Explained variance by component:")
    for i, variance_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"Component {i+1}: {variance_ratio:.2%}")

    # Create a PCA plot colored by each numerical column
    for column in numerical_cols:
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(principalDf['PC1'], principalDf['PC2'], c=df[column], cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label=column)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'PCA of Dataset colored by {column} (PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%})')
        plt.grid(True)
        plt.savefig(f"{output_pca_image_base}_{column}.png")
        plt.close()  # Close the figure to free memory

# Example usage
create_pca_plots('merged.csv', 'pca_plot')
