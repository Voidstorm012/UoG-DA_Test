"""
Simplified UWB LOS/NLOS Classification and Range Prediction Project

This script performs a basic analysis of the UWB dataset:
1. Load the dataset
2. Train a simple classifier for LOS/NLOS classification
3. Train a simple regressor for range prediction
4. Evaluate and visualize results
"""
import os
import sys
import time
import random

# Try to import required packages
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                                mean_squared_error, mean_absolute_error, r2_score)
    HAVE_PACKAGES = True
except ImportError:
    print("Some packages are missing. Running in limited mode.")
    HAVE_PACKAGES = False

def load_dataset():
    """Load UWB dataset directly from CSV files."""
    print("Loading UWB dataset...")
    
    dataset_dir = os.path.join(os.path.dirname(__file__), 'data', 'dataset')
    print(f"Looking for dataset in: {dataset_dir}")
    
    all_data = []
    feature_names = None
    
    # Check each CSV file
    for i in range(1, 8):
        filename = f"uwb_dataset_part{i}.csv"
        filepath = os.path.join(dataset_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Loading {filename}...")
            try:
                # Read CSV file
                df = pd.read_csv(filepath)
                
                # Save feature names from first file
                if feature_names is None:
                    feature_names = df.columns.tolist()
                
                # Add data to our collection
                all_data.append(df)
                print(f"  Loaded {len(df)} samples")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
        else:
            print(f"File not found: {filepath}")
    
    # Combine all dataframes
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total samples loaded: {len(combined_df)}")
        
        # Get class distribution
        class_dist = combined_df.iloc[:, 0].value_counts()
        print(f"Class distribution:\n{class_dist}")
        
        return combined_df, feature_names
    else:
        print("No data loaded!")
        return None, None

def simple_analysis(data, feature_names):
    """Perform basic analysis on the dataset."""
    print("\nPerforming basic analysis...")
    
    # Get basic statistics for each feature
    print("\nFeature Statistics:")
    stats = data.describe()
    print(stats)
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    if HAVE_PACKAGES:
        try:
            # Plot class distribution
            plt.figure(figsize=(8, 5))
            ax = data.iloc[:, 0].value_counts().plot(kind='bar')
            plt.title('Class Distribution (0=LOS, 1=NLOS)')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
            plt.close()
            
            # Plot distributions of key features
            for i, feature in enumerate(feature_names[1:15]):  # Skip class column and CIR values
                plt.figure(figsize=(10, 6))
                
                # Split by class
                los_data = data[data.iloc[:, 0] == 0][feature]
                nlos_data = data[data.iloc[:, 0] == 1][feature]
                
                plt.hist([los_data, nlos_data], bins=30, alpha=0.6, 
                         label=['LOS', 'NLOS'])
                
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'feature_dist_{feature}.png'))
                plt.close()
                
                print(f"Created plot for {feature}")
            
            print(f"All plots saved to {plots_dir}")
        except Exception as e:
            print(f"Error creating plots: {e}")

def train_simple_model(data):
    """Train a simple classifier on the dataset."""
    if not HAVE_PACKAGES:
        print("\nSkipping model training (required packages not available)")
        return
    
    print("\nTraining a simple Random Forest classifier...")
    
    # Prepare data
    X = data.iloc[:, 1:15]  # Features (without CIR)
    y = data.iloc[:, 0]     # Target (LOS/NLOS)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Feature importance
    print("\nFeature Importance:")
    importance = model.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['LOS (0)', 'NLOS (1)'])
    plt.yticks(tick_marks, ['LOS (0)', 'NLOS (1)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Add text to confusion matrix cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"Model evaluation plots saved to {plots_dir}")

def train_range_prediction_model(data):
    """Train a simple regression model for range prediction."""
    if not HAVE_PACKAGES:
        print("\nSkipping range prediction model training (required packages not available)")
        return
    
    print("\nTraining a simple Random Forest regressor for range prediction...")
    
    # Prepare data
    # For range prediction, we use the 'Range' column as target and include NLOS flag as a feature
    X = pd.concat([data.iloc[:, 0:1], data.iloc[:, 2:15]], axis=1)  # NLOS flag + other features (without Range and CIR)
    y = data.iloc[:, 1]  # Target (Range)
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train Random Forest regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Training model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    print("\nFeature Importance for Range Prediction:")
    importance = model.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importance[idx]:.4f}")
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importance for Range Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance_range.png'))
    plt.close()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.3)
    
    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Range')
    plt.xlabel('Actual Range')
    plt.ylabel('Predicted Range')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted_range.png'))
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 8))
    
    # Histogram of residuals
    plt.subplot(2, 1, 1)
    plt.hist(residuals, bins=30)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    # Scatter plot of residuals vs predicted
    plt.subplot(2, 1, 2)
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Range')
    plt.ylabel('Residual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'residuals_range.png'))
    plt.close()
    
    print(f"Range prediction model plots saved to {plots_dir}")

def main():
    """Main function to run the analysis."""
    print("=" * 80)
    print("UWB LOS/NLOS Classification and Range Prediction - Simplified Analysis")
    print("=" * 80)
    
    # Load dataset
    data, feature_names = load_dataset()
    
    if data is not None:
        # Basic analysis
        simple_analysis(data, feature_names)
        
        # Train classification model
        print("\n" + "=" * 50)
        print("TASK 1: LOS/NLOS CLASSIFICATION")
        print("=" * 50)
        train_simple_model(data)
        
        # Train regression model
        print("\n" + "=" * 50)
        print("TASK 2: RANGE PREDICTION")
        print("=" * 50)
        train_range_prediction_model(data)
        
        print("\nAnalysis completed successfully!")
    else:
        print("\nAnalysis failed: Could not load dataset")

if __name__ == "__main__":
    main()