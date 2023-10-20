import pandas as pd
import matplotlib.pyplot as plt
import os

# Function to create boxplots and save image
def create_and_save_boxplot(csv_file_path, output_image_path):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)

    # targetColumnName = 'Test Kendall'
    targetColumnName = 'Train Kendall'
    if targetColumnName in data.columns:
        # Group the data by features and calculate median for each group
        grouped_data = data.groupby('Features')[targetColumnName].apply(list)
        medians = grouped_data.apply(lambda x: pd.Series(x).median())

        # Sort the groups by median values
        sorted_groups = medians.sort_values(ascending=False).index
        sorted_data = grouped_data[sorted_groups]

        # Create a dictionary to store colors based on string characteristics
        color_dict = {
            'T': 'blue',          # Color for strings containing 'T'
            'noT': 'orange',      # Color for strings not containing 'T'
            'single': 'green'     # Color for strings with a single letter
        }

        # Determine colors based on string characteristics
        colors = []
        for feature in sorted_groups:
            if 'T' in feature:
                colors.append(color_dict['T'])
            elif len(set(feature)) == 1:
                colors.append(color_dict['single'])
            else:
                colors.append(color_dict['noT'])

        # Create boxplots for the sorted groups with custom colors
        plt.figure(figsize=(10, 10))
        boxplot = plt.boxplot(sorted_data.values, patch_artist=True)

        # Apply the colors to boxplots
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)

        # Print the labels of the horizontal axis to the console
        x_labels = sorted_data.index
        print("Labels of the Horizontal Axis:")
        for label in x_labels:
            print(label)
        
        plt.xticks(range(1, len(sorted_data) + 1), sorted_data.index, rotation=45)
        plt.xlabel('Features')
        plt.ylabel( targetColumnName+' Values')
        plt.title('Boxplots for Features (Sorted by Median)')

        # Set the y-axis limits
        plt.ylim(-0.10, 0.30)
        #plt.ylim(0.10, 0.30)

        # Create a legend with labels for the colors inside the image
        legend_labels = ['Combination including Type', 'Combination not including Type', 'Single feature combination']
        legend_colors = [color_dict['T'], color_dict['noT'], color_dict['single']]
        legend_patches = [plt.Line2D([0], [0], marker='s', color=color, label=label, markersize=10) for color, label in zip(legend_colors, legend_labels)]
        plt.legend(handles=legend_patches, loc='upper right')

        # Save the image
        plt.savefig(output_image_path)
        plt.close()
    else:
        print( targetColumnName+" column does not exist in "+csv_file_path+". Change the code from Test Kendall to Train Kendall column?")

# File paths and output image paths
file_paths = ['nangateSTD-onlyIO-ablationResult.csv']
output_image_paths = [os.path.splitext(filename)[0] + '.png' for filename in file_paths]

# Generate boxplots and save images for each file
for file_path, output_image_path in zip(file_paths, output_image_paths):
    create_and_save_boxplot(file_path, output_image_path)
