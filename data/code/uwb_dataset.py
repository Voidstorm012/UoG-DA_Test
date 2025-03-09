"""
Created on Feb 6, 2017

@author: Klemen Bregar 
"""

import os
import pandas as pd
from numpy import vstack


def import_from_files():
    """
        Read .csv files and store data into an array
        format: |LOS|NLOS|data...|
    """
    # Use a path relative to the current directory, or try a few different options
    possible_paths = [
        'dataset/',  # Direct subdirectory
        '../dataset/',  # One level up
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset')  # Absolute path
    ]
    
    output_arr = []
    first = 1
    rootdir = None
    
    # Find the first working path
    for path in possible_paths:
        if os.path.exists(path):
            rootdir = path
            break
    
    # Check if we found a valid dataset directory
    if rootdir is None:
        print("Dataset directory not found. Tried the following paths:")
        for path in possible_paths:
            print(f"- {os.path.abspath(path)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Available directories: {os.listdir('.')}")
        return output_arr
    
    print(f"Loading files from: {os.path.abspath(rootdir)}")
    
    # List all CSV files in the dataset directory
    csv_files = [f for f in os.listdir(rootdir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    if len(csv_files) == 0:
        print("No CSV files found in the dataset directory")
        return output_arr
    
    for filename in sorted(csv_files):
        filepath = os.path.join(rootdir, filename)
        print(f"Loading: {filepath}")
        
        try:
            # Read data from file
            df = pd.read_csv(filepath, sep=',', header=0)
            
            # Convert to numpy array (using modern method)
            input_data = df.values
            
            # Append to output array
            if first > 0:
                first = 0
                output_arr = input_data
            else:
                output_arr = vstack((output_arr, input_data))
                
            print(f"  Loaded {len(input_data)} samples from {filename}")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    
    if len(output_arr) > 0:
        print(f"Total samples loaded: {len(output_arr)}")
    else:
        print("No data loaded")
    
    return output_arr

if __name__ == '__main__':

    # import raw data from folder with dataset
    print("Importing dataset to numpy array")
    print("-------------------------------")
    data = import_from_files()
    print("-------------------------------")
    # print dimensions and data
    print("Number of samples in dataset: %d" % len(data))
    print("Length of one sample: %d" % len(data[0]))
    print("-------------------------------")
    print("Dataset:")
    print(data)