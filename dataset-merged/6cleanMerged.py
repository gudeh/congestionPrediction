import pandas as pd

def drop_and_write_csv(csv_file_path, output_file):
    # Specify the data types for each column
    dtype_spec = {
        'name': 'object',
        'type': 'object',
        'logic_function': 'object',
        'Dataset': 'object',
        'Design': 'object'
    }
    
    # Load data from CSV with specified data types
    df = pd.read_csv(csv_file_path, dtype=dtype_spec, low_memory=False)
    print("Initial data shape:", df.shape)

    # Filter out rows where the 'type' column contains 'fakeram'
    df = df[~df['type'].str.contains('fakeram', na=False)]
    print("After filtering 'fakeram':", df.shape)

    # Columns to exclude (uncomment if needed)
    # columns_to_exclude = ['name', 'type', 'logic_function', 'Dataset', 'Design']
    # df.drop(columns=columns_to_exclude, inplace=True, errors='ignore')
    # print("After excluding specified columns:", df.shape)

    # Convert specific columns that should be numeric
    numeric_cols = ['routingHeat', 'area', 'input_pins', 'output_pins', 'closeness', 'harmonic', 'percolation', 'load', 'betweenness', 'Eigen', 'pageRank', 'inDegree', 'outDegree']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("After converting to numeric:", df.shape)

    # Drop any remaining NaNs
    df.dropna(inplace=True)
    print("After dropping NaNs:", df.shape)

    # Ensure the output file has a .csv extension
    if not output_file.endswith('.csv'):
        output_file += '.csv'

    # Write the cleaned DataFrame to a new CSV file with float values formatted to 2 decimal places
    df.to_csv(output_file, index=False, float_format='%.2f')
    print("Final data shape written to CSV:", df.shape)

# Example usage
drop_and_write_csv('merged.csv', 'merged-cleaner.csv')
