"""
UWB LOS/NLOS Classification and Range Prediction Project
CSC3105 Mini Project

This script orchestrates the complete data analytics pipeline:
1. Data Preparation
2. Data Mining
3. Data Visualization
4. Results Analysis

The project has two main objectives:
1. LOS/NLOS Classification: Identify whether a signal path is Line-of-Sight or Non-Line-of-Sight
2. Range Prediction: Estimate the measured range based on signal characteristics
"""
import os
import sys
import time
import argparse

# Add the UWB dataset directory to the path
UWB_DATASET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'UWB-LOS-NLOS-Data-Set'))
sys.path.append(UWB_DATASET_PATH)

# Import our modules
from data_preparation import preprocess
from data_mining import models
from data_mining import regression_models
from data_visualization import visualize
from results import analyze

def run_classification_pipeline():
    """Run the LOS/NLOS classification pipeline"""
    print("=" * 80)
    print("UWB LOS/NLOS Classification Pipeline")
    print("=" * 80)
    
    # Timer for performance tracking
    start_time = time.time()
    
    # Step 1: Data Preparation
    print("\n[1/4] Data Preparation")
    print("-" * 40)
    X_train, X_test, y_train, y_test, feature_names, raw_data, processed_data = preprocess.prepare_data('classification')
    
    # Step 2: Data Mining
    print("\n[2/4] Data Mining")
    print("-" * 40)
    trained_models = models.train_models(X_train, y_train)
    model_results = models.evaluate_models(trained_models, X_test, y_test)
    
    # Step 3: Data Visualization
    print("\n[3/4] Data Visualization")
    print("-" * 40)
    visualize.plot_results(model_results, X_train, y_train, X_test, y_test, feature_names, 'classification', 
                          raw_data=raw_data, processed_data=processed_data)
    
    # Step 4: Results Analysis
    print("\n[4/4] Results Analysis")
    print("-" * 40)
    analyze.interpret_results(model_results, feature_names, 'classification')
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    print("\nClassification pipeline completed successfully!")

def run_regression_pipeline():
    """Run the range prediction pipeline"""
    print("=" * 80)
    print("UWB Range Prediction Pipeline")
    print("=" * 80)
    
    # Timer for performance tracking
    start_time = time.time()
    
    # Step 1: Data Preparation
    print("\n[1/4] Data Preparation")
    print("-" * 40)
    X_train, X_test, y_train, y_test, feature_names, raw_data, processed_data = preprocess.prepare_data('regression')
    
    # Step 2: Data Mining
    print("\n[2/4] Data Mining")
    print("-" * 40)
    trained_models = regression_models.train_regression_models(X_train, y_train)
    model_results = regression_models.evaluate_regression_models(trained_models, X_test, y_test)
    
    # Step 3: Data Visualization
    print("\n[3/4] Data Visualization")
    print("-" * 40)
    visualize.plot_results(model_results, X_train, y_train, X_test, y_test, feature_names, 'regression',
                          raw_data=raw_data, processed_data=processed_data)
    
    # Step 4: Results Analysis
    print("\n[4/4] Results Analysis")
    print("-" * 40)
    analyze.interpret_results(model_results, feature_names, 'regression')
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    print("\nRegression pipeline completed successfully!")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='UWB LOS/NLOS Classification and Range Prediction')
    parser.add_argument('--task', type=str, default='both',
                        choices=['classification', 'regression', 'both'],
                        help='Task to run: classification, regression, or both')
    args = parser.parse_args()
    
    # Run the selected task(s)
    if args.task in ['classification', 'both']:
        run_classification_pipeline()
    
    if args.task in ['regression', 'both']:
        # If running both, add a separator
        if args.task == 'both':
            print("\n\n" + "=" * 80)
        run_regression_pipeline()
    
    if args.task == 'both':
        print("\n" + "=" * 80)
        print("Both classification and regression pipelines completed successfully!")

if __name__ == "__main__":
    main()