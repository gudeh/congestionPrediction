import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

def create_correlation_and_pca_plots(csv_file_path, output_correlation_image, output_pca_image):
    # Load data from CSV
    df = pd.read_csv(csv_file_path)

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

    # Standardizing the features for PCA
    features_scaled = df[numerical_cols] #StandardScaler().fit_transform(df[numerical_cols])

    # PCA
    pca = PCA(n_components=2)  # Reduce to 2 components
    principalComponents = pca.fit_transform(features_scaled)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    # Save PCA plot to file
    plt.figure(figsize=(8, 6))
    plt.scatter(principalDf['PC1'], principalDf['PC2'], alpha=0.7)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Dataset')
    plt.grid(True)
    plt.savefig(output_pca_image)
    plt.close()  # Close the figure to free memory

# Example usage
csv_file_path = 'merged.csv'
output_correlation_image = 'correlation_heatmap.png'
output_pca_image = 'noScalr-pca_plot.png'
create_correlation_and_pca_plots(csv_file_path, output_correlation_image, output_pca_image)
