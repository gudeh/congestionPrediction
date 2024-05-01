import os
import shutil

def copy_data_files(source_root, target_root):
    # Walk through the source directory
    for subdir, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith("-data.csv"):
                source_path = os.path.join(subdir, file)
                
                # Extract the relative path
                rel_dir = os.path.relpath(subdir, source_root)
                
                # Prepare the target directory path by modifying the source relative path
                # Assuming the target directory structure has a prefix pattern in folder name
                target_dir = os.path.join(target_root, rel_dir.replace('+ID|OD|E|PG', ''))
                
                # Ensure the target directory exists
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                # Prepare the target file path
                target_path = os.path.join(target_dir, file)
                
                # Copy the file to the target directory
                shutil.copy(source_path, target_path)
                print(f"Copied '{source_path}' to '{target_path}'")

# Set your directories here
source_root = '/home/gudeh/Desktop/toCopy/'  # e.g., '/home/username/Desktop/toCopy'
target_root = '/home/gudeh/Desktop/tese-dataset-merged'  # e.g., '/home/username/Desktop/tese-dataset-merged'

copy_data_files(source_root, target_root)
