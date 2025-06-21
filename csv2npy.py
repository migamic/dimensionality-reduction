import numpy as np
import pandas as pd
import os
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_file> <output_directory> <target_column>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_directory = sys.argv[2]
    target_column = sys.argv[3]
    
    data = pd.read_csv(csv_file)
    
    if target_column not in data.columns:
        print(f"Error: Target column '{target_column}' not found in CSV file.")
        sys.exit(1)
    
    X = data.drop(columns=[target_column]).values
    y = data[target_column].values
    
    # Ensure that X is of numeric type (float64)
    X = X.astype(np.float64)
    
    if y.dtype == 'object':  # If y contains strings (categories)
        y, _ = pd.factorize(y)  # Convert strings to unique integers
    
    y = y.astype(np.int64)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    np.save(os.path.join(output_directory, 'X.npy'), X)
    np.save(os.path.join(output_directory, 'y.npy'), y)
    
    print(f"Data saved in '{output_directory}' as X.npy and y.npy")

if __name__ == "__main__":
    main()

