"""
Data Preparation Module

This module handles:
1. Loading the UWB dataset
2. Cleaning and preprocessing data
3. Feature selection and engineering
4. Train-test splitting
"""
import os
import sys
import random
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy not available. Using basic Python lists.")
    NUMPY_AVAILABLE = False

# Use our local copy of the dataset module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'code')))

# Import UWB dataset module
try:
    import uwb_dataset
    print("Successfully imported UWB dataset module")
except ImportError as e:
    print(f"Error importing UWB dataset module: {e}")
    # Fallback import method
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            "uwb_dataset", 
            os.path.abspath(os.path.join('..', 'data', 'code', 'uwb_dataset.py'))
        )
        uwb_dataset = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(uwb_dataset)
        print("Imported UWB dataset module using importlib")
    except Exception as e:
        print(f"Failed to import using importlib: {e}")

# Feature names
FEATURE_NAMES = [
    "NLOS",
    "Range",
    "FP_IDX",
    "FP_AMP1",
    "FP_AMP2",
    "FP_AMP3",
    "STDEV_NOISE",
    "CIR_PWR",
    "MAX_NOISE",
    "RXPACC",
    "CH",
    "FRAME_LEN",
    "PREAM_LEN",
    "BITRATE",
    "PRFR"
]

def load_data():
    """
    Load the UWB dataset.
    
    Returns:
        data: Numpy array with all dataset samples
    """
    print("Loading UWB dataset...")
    
    # Save the original working directory to restore it later
    original_dir = os.getcwd()
    
    try:
        # Change to our data directory to ensure relative paths work correctly
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        os.chdir(data_dir)
        
        # Load the dataset using the module function
        data = uwb_dataset.import_from_files()
        print(f"Dataset loaded: {len(data)} samples with {len(data[0])} features")
        
        # Check for NaN values
        try:
            if np.isnan(data).any():
                print("Warning: Dataset contains NaN values")
        except:
            print("Warning: NumPy not available for NaN check")
    
    finally:
        # Restore the original working directory
        os.chdir(original_dir)
    
    return data

def normalize_cir(data):
    """
    Normalize CIR values by RX preamble count.
    
    Args:
        data: Dataset with CIR values
    
    Returns:
        data: Dataset with normalized CIR values
    """
    print("Normalizing CIR values...")
    for item in data:
        item[15:] = item[15:]/float(item[9])  # RXPACC is at index 9
    
    return data

def split_features_target(data, target_type='classification'):
    """
    Split dataset into features (X) and target (y).
    
    Args:
        data: Full dataset array
        target_type: Type of target ('classification' for LOS/NLOS or 'regression' for range)
    
    Returns:
        X: Features matrix
        y: Target vector
    """
    if target_type == 'classification':
        # First column is the target (NLOS/LOS)
        y = data[:, 0]
        
        # Metadata features (without CIR values and without Range for classification)
        X = data[:, 2:15]  # Skip NLOS flag (0) and Range (1)
    
    elif target_type == 'regression':
        # For range prediction, the Range is our target (column 1)
        y = data[:, 1]
        
        # For regression, we use NLOS flag as a feature along with other metadata
        # Note: we exclude the Range (target) from features
        X = np.hstack((data[:, 0:1], data[:, 2:15]))  # NLOS flag and other metadata
    
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    return X, y

def feature_engineering(X):
    """
    Perform feature engineering to create new features.
    
    Args:
        X: Original features matrix
    
    Returns:
        X_new: Features matrix with engineered features
    """
    # For now, we're just returning the original features
    # This function will be expanded later based on feature importance analysis
    return X

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Features matrix
        y: Target vector
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    # Set random seed for reproducibility
    random.seed(random_state)
    try:
        np.random.seed(random_state)
    except:
        pass
    
    # Get indices and shuffle them
    indices = list(range(len(y)))
    random.shuffle(indices)
    
    # Split indices
    test_count = int(len(indices) * test_size)
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Split the data
    try:
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
    except:
        # Fallback to list comprehension if numpy indexing fails
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
    
    print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    return X_train, X_test, y_train, y_test

def prepare_data(target_type='classification'):
    """
    Main function that orchestrates data preparation pipeline.
    
    Args:
        target_type: Type of target ('classification' for LOS/NLOS or 'regression' for range)
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, raw_data, processed_data: Prepared data for modeling
    """
    # Load raw data
    raw_data = load_data()
    
    # Keep a copy of the raw data for visualization
    raw_data_copy = raw_data.copy() if NUMPY_AVAILABLE else [item[:] for item in raw_data]
    
    # Normalize CIR values
    processed_data = normalize_cir(raw_data)
    
    # Keep a copy of the processed data for visualization
    processed_data_copy = processed_data.copy() if NUMPY_AVAILABLE else [item[:] for item in processed_data]
    
    # Split features and target based on target type
    X, y = split_features_target(processed_data, target_type)
    
    # Feature engineering
    X = feature_engineering(X)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Return prepared data with appropriate feature names
    if target_type == 'classification':
        return X_train, X_test, y_train, y_test, FEATURE_NAMES[2:15], raw_data_copy, processed_data_copy  # Skip NLOS and Range
    else:  # regression
        # For regression, NLOS flag is a feature
        return X_train, X_test, y_train, y_test, [FEATURE_NAMES[0]] + FEATURE_NAMES[2:15], raw_data_copy, processed_data_copy  # NLOS flag and other metadata