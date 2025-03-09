"""
Data Visualization Module

This module handles:
1. Plotting model performance metrics
2. Creating confusion matrices
3. Visualizing feature importance
4. Generating ROC curves
5. Visualizing regression results
"""
import os
import time
import math

# Try to import pyplot and handle gracefully if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: Matplotlib not available. Visualizations will be skipped.")
    MATPLOTLIB_AVAILABLE = False

# Try to import numpy and handle gracefully if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy not available. Visualizations will be skipped.")
    NUMPY_AVAILABLE = False

# Try to import seaborn for enhanced visualizations
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    print("Warning: Seaborn not available. Using basic matplotlib.")
    SEABORN_AVAILABLE = False

# Try to import pandas for data manipulation
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Warning: Pandas not available. Some visualizations will be simplified.")
    PANDAS_AVAILABLE = False

# Create output directories for plots
PLOT_DIR = os.path.join('..', 'results', 'plots')
PREPROC_PLOT_DIR = os.path.join(PLOT_DIR, 'preprocessing')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(PREPROC_PLOT_DIR, exist_ok=True)

def plot_model_comparison(model_results, metric='accuracy'):
    """
    Plot comparison of different models on a specific metric.
    
    Args:
        model_results: Dictionary of model evaluation results
        metric: Metric to compare (accuracy, precision, recall, f1_score)
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping model comparison plot for {metric}")
        return
    
    # Extract metric values
    models = []
    values = []
    
    for model_name, results in model_results.items():
        if metric in results:
            models.append(model_name)
            values.append(results[metric])
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(models, values, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison: {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'model_comparison_{metric}.png'))
    plt.close()
    
    print(f"Model comparison plot for {metric} saved to {PLOT_DIR}")

def plot_confusion_matrix(model_results, model_name):
    """
    Plot confusion matrix for a specific model.
    
    Args:
        model_results: Dictionary of model evaluation results
        model_name: Name of the model to plot
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping confusion matrix plot for {model_name}")
        return
    
    # Check if confusion matrix is available
    if model_name not in model_results or 'confusion_matrix' not in model_results[model_name]:
        print(f"Confusion matrix not available for {model_name}")
        return
    
    # Get confusion matrix
    cm = model_results[model_name]['confusion_matrix']
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar()
    
    # Add labels
    classes = ['LOS (0)', 'NLOS (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    print(f"Confusion matrix plot for {model_name} saved to {PLOT_DIR}")

def plot_feature_importance(model_results, model_name, feature_names):
    """
    Plot feature importance for a specific model.
    
    Args:
        model_results: Dictionary of model evaluation results
        model_name: Name of the model to plot
        feature_names: List of feature names
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping feature importance plot for {model_name}")
        return
    
    # Check if feature importance is available
    if model_name not in model_results or 'feature_importance' not in model_results[model_name]:
        print(f"Feature importance not available for {model_name}")
        return
    
    # Get feature importance
    importance = model_results[model_name]['feature_importance']
    
    # Skip if feature importance is not available for this model
    if importance is None:
        print(f"  Feature importance not available for {model_name}")
        return
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance
    indices = np.argsort(importance)
    
    plt.barh(range(len(indices)), importance[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title(f'Feature Importance: {model_name}')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'feature_importance_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    print(f"Feature importance plot for {model_name} saved to {PLOT_DIR}")

def plot_roc_curve(model_results, model_name):
    """
    Plot ROC curve for a specific model.
    
    Args:
        model_results: Dictionary of model evaluation results
        model_name: Name of the model to plot
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping ROC curve plot for {model_name}")
        return
    
    # Check if ROC curve data is available
    if (model_name not in model_results or 
        'fpr' not in model_results[model_name] or 
        'tpr' not in model_results[model_name] or
        'roc_auc' not in model_results[model_name]):
        print(f"ROC curve data not available for {model_name}")
        return
    
    # Get ROC curve data
    fpr = model_results[model_name]['fpr']
    tpr = model_results[model_name]['tpr']
    roc_auc = model_results[model_name]['roc_auc']
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic: {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'roc_curve_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    print(f"ROC curve plot for {model_name} saved to {PLOT_DIR}")

def plot_learning_curve(X_train, y_train, model_name, model):
    """
    Plot learning curve for a specific model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_name: Name of the model
        model: Trained model object
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping learning curve plot for {model_name}")
        return
    
    try:
        from sklearn.model_selection import learning_curve
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Calculate learning curve
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes, cv=5, 
            scoring='accuracy', n_jobs=-1)
        
        # Calculate mean and standard deviation for training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        
        # Calculate mean and standard deviation for test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        
        # Plot bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        
        # Add labels and title
        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title(f"Learning Curve: {model_name}")
        plt.legend(loc="best")
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(PLOT_DIR, f'learning_curve_{model_name.replace(" ", "_")}.png'))
        plt.close()
        
        print(f"Learning curve plot for {model_name} saved to {PLOT_DIR}")
    except ImportError:
        print(f"Skipping learning curve plot for {model_name} (scikit-learn not available)")

# New functions for regression visualization
def plot_actual_vs_predicted(model_results, model_name, X_test, y_test):
    """
    Plot actual vs predicted values for a regression model.
    
    Args:
        model_results: Dictionary of model evaluation results
        model_name: Name of the model to plot
        X_test: Test features
        y_test: Actual range values
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping actual vs predicted plot for {model_name}")
        return
    
    # Check if model is available
    if model_name not in model_results or 'model' not in model_results[model_name]:
        print(f"Model not available for {model_name}")
        return
    
    # Get model and make predictions
    model = model_results[model_name]['model']
    scaler = model_results[model_name].get('scaler')
    
    # Scale features if needed
    X_test_scaled = X_test
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.3)
    
    # Add perfect prediction line
    min_val = min(np.min(y_test), np.min(y_pred))
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted Range: {model_name}')
    plt.xlabel('Actual Range')
    plt.ylabel('Predicted Range')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'range_prediction_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    print(f"Actual vs predicted plot for {model_name} saved to {PLOT_DIR}")

def plot_residuals(model_results, model_name, X_test, y_test):
    """
    Plot residuals for a regression model.
    
    Args:
        model_results: Dictionary of model evaluation results
        model_name: Name of the model to plot
        X_test: Test features
        y_test: Actual range values
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping residuals plot for {model_name}")
        return
    
    # Check if model is available
    if model_name not in model_results or 'model' not in model_results[model_name]:
        print(f"Model not available for {model_name}")
        return
    
    # Get model and make predictions
    model = model_results[model_name]['model']
    scaler = model_results[model_name].get('scaler')
    
    # Scale features if needed
    X_test_scaled = X_test
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Residuals histogram
    plt.subplot(2, 1, 1)
    plt.hist(residuals, bins=30)
    plt.title(f'Histogram of Residuals: {model_name}')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    # Residuals vs predicted
    plt.subplot(2, 1, 2)
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals vs Predicted Values: {model_name}')
    plt.xlabel('Predicted Range')
    plt.ylabel('Residual')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'residuals_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    print(f"Residuals plot for {model_name} saved to {PLOT_DIR}")

def plot_regression_comparison(model_results, metric='rmse'):
    """
    Plot comparison of different regression models on a specific metric.
    
    Args:
        model_results: Dictionary of model evaluation results
        metric: Metric to compare (rmse, mae, r2)
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print(f"Skipping regression model comparison plot for {metric}")
        return
    
    # Extract metric values
    models = []
    values = []
    
    for model_name, results in model_results.items():
        if metric in results:
            models.append(model_name)
            values.append(results[metric])
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='skyblue')
    plt.xlabel('Model')
    plt.ylabel(metric.upper() if metric != 'r2' else 'R²')
    plt.title(f'Regression Model Comparison: {metric.upper() if metric != "r2" else "R²"}')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(PLOT_DIR, f'regression_comparison_{metric}.png'))
    plt.close()
    
    print(f"Regression model comparison plot for {metric} saved to {PLOT_DIR}")

def visualize_data_distribution(data, feature_names, stage="raw", is_normalized=False):
    """
    Visualize the distribution of features in the dataset before/after preprocessing.
    
    Args:
        data: Raw dataset array
        feature_names: List of feature names
        stage: Stage of processing ("raw" or "processed")
        is_normalized: Whether the data has been normalized
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print("Skipping data distribution visualization due to missing dependencies.")
        return
    
    print(f"Visualizing {stage} data distribution...")
    
    # Plot class distribution (first column is NLOS/LOS)
    plt.figure(figsize=(10, 6))
    classes, counts = np.unique(data[:, 0], return_counts=True)
    class_labels = ['LOS (0)', 'NLOS (1)'] if len(classes) == 2 else [f'Class {int(c)}' for c in classes]
    
    plt.bar(class_labels, counts, color=['green', 'red'] if len(classes) == 2 else None)
    plt.title(f'Class Distribution ({stage} data)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(PREPROC_PLOT_DIR, f'class_distribution_{stage}.png'))
    plt.close()
    
    # Plot feature distributions (metadata features 1-14)
    # Create a grid of plots for the features
    n_features = min(len(feature_names), 13)  # Skip the CIR values, just plot metadata
    n_cols = 3
    n_rows = math.ceil(n_features / n_cols)
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, feature_idx in enumerate(range(1, n_features + 1)):  # Skip NLOS flag (0)
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Get feature data and split by class
        feature_data = data[:, feature_idx]
        los_mask = data[:, 0] == 0
        nlos_mask = data[:, 0] == 1
        
        # Plot histograms by class
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {feature_idx}'
        if SEABORN_AVAILABLE:
            sns.histplot(feature_data[los_mask], color='green', alpha=0.5, label='LOS', kde=True, ax=ax)
            sns.histplot(feature_data[nlos_mask], color='red', alpha=0.5, label='NLOS', kde=True, ax=ax)
        else:
            plt.hist(feature_data[los_mask], bins=30, color='green', alpha=0.5, label='LOS')
            plt.hist(feature_data[nlos_mask], bins=30, color='red', alpha=0.5, label='NLOS')
        
        plt.title(f'{feature_name} Distribution')
        plt.xlabel(feature_name + (' (Normalized)' if is_normalized else ''))
        plt.ylabel('Frequency')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PREPROC_PLOT_DIR, f'feature_distributions_{stage}.png'))
    plt.close()
    
    # Create individual feature distribution plots for more detail
    for i, feature_idx in enumerate(range(1, n_features + 1)):
        plt.figure(figsize=(10, 6))
        
        # Get feature data and split by class
        feature_data = data[:, feature_idx]
        los_mask = data[:, 0] == 0
        nlos_mask = data[:, 0] == 1
        
        # Plot histograms by class
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {feature_idx}'
        if SEABORN_AVAILABLE:
            sns.histplot(feature_data[los_mask], color='green', alpha=0.5, label='LOS', kde=True)
            sns.histplot(feature_data[nlos_mask], color='red', alpha=0.5, label='NLOS', kde=True)
        else:
            plt.hist(feature_data[los_mask], bins=30, color='green', alpha=0.5, label='LOS')
            plt.hist(feature_data[nlos_mask], bins=30, color='red', alpha=0.5, label='NLOS')
        
        plt.title(f'{feature_name} Distribution')
        plt.xlabel(feature_name + (' (Normalized)' if is_normalized else ''))
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PREPROC_PLOT_DIR, f'feature_dist_{feature_name.replace(" ", "_")}_{stage}.png'))
        plt.close()

def visualize_cir_patterns(data, sample_indices=None, stage="raw"):
    """
    Visualize Channel Impulse Response (CIR) patterns for LOS and NLOS.
    
    Args:
        data: Dataset array
        sample_indices: Indices of samples to visualize (randomly selected if None)
        stage: Stage of processing ("raw" or "processed")
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print("Skipping CIR visualization due to missing dependencies.")
        return
    
    print(f"Visualizing CIR patterns ({stage} data)...")
    
    # If no indices provided, randomly select samples
    if sample_indices is None:
        los_indices = np.where(data[:, 0] == 0)[0]
        nlos_indices = np.where(data[:, 0] == 1)[0]
        
        np.random.seed(42)  # For reproducibility
        los_samples = np.random.choice(los_indices, min(5, len(los_indices)), replace=False)
        nlos_samples = np.random.choice(nlos_indices, min(5, len(nlos_indices)), replace=False)
        sample_indices = {'LOS': los_samples, 'NLOS': nlos_samples}
    
    # CIR values start from index 15
    cir_start_idx = 15
    
    # Plot CIR patterns
    plt.figure(figsize=(15, 10))
    
    # LOS samples
    plt.subplot(2, 1, 1)
    for idx in sample_indices['LOS']:
        cir_values = data[idx, cir_start_idx:]
        plt.plot(cir_values, alpha=0.7, label=f'Sample {idx}')
    
    plt.title('LOS CIR Patterns')
    plt.xlabel('Sample Index')
    plt.ylabel('CIR Value' + (' (Normalized)' if stage == "processed" else ''))
    plt.grid(linestyle='--', alpha=0.7)
    if len(sample_indices['LOS']) <= 10:  # Only show legend if not too many samples
        plt.legend()
    
    # NLOS samples
    plt.subplot(2, 1, 2)
    for idx in sample_indices['NLOS']:
        cir_values = data[idx, cir_start_idx:]
        plt.plot(cir_values, alpha=0.7, label=f'Sample {idx}')
    
    plt.title('NLOS CIR Patterns')
    plt.xlabel('Sample Index')
    plt.ylabel('CIR Value' + (' (Normalized)' if stage == "processed" else ''))
    plt.grid(linestyle='--', alpha=0.7)
    if len(sample_indices['NLOS']) <= 10:  # Only show legend if not too many samples
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PREPROC_PLOT_DIR, f'cir_patterns_{stage}.png'))
    plt.close()

def visualize_feature_correlations(data, feature_names, stage="raw"):
    """
    Visualize correlations between features.
    
    Args:
        data: Dataset array
        feature_names: List of feature names
        stage: Stage of processing ("raw" or "processed")
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print("Skipping correlation visualization due to missing dependencies.")
        return
    
    if not SEABORN_AVAILABLE or not PANDAS_AVAILABLE:
        print("Skipping correlation visualization due to missing seaborn or pandas.")
        return
    
    print(f"Visualizing feature correlations ({stage} data)...")
    
    # Extract metadata features (skip CIR values)
    n_features = min(len(feature_names) + 1, 15)  # Include NLOS flag (0) and metadata features
    feature_data = data[:, :n_features]
    
    # Create a DataFrame for correlation analysis
    column_names = ['NLOS'] + feature_names[:n_features-1]
    df = pd.DataFrame(feature_data, columns=column_names)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=.5, vmin=-1, vmax=1)
    plt.title(f'Feature Correlation Matrix ({stage} data)')
    plt.tight_layout()
    plt.savefig(os.path.join(PREPROC_PLOT_DIR, f'feature_correlations_{stage}.png'))
    plt.close()

def plot_results(model_results, X_train, y_train, X_test, y_test, feature_names, task_type='classification', raw_data=None, processed_data=None):
    """
    Create all visualization plots for model evaluation.
    
    Args:
        model_results: Dictionary of model evaluation results
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
        feature_names: List of feature names
        task_type: Type of task ('classification' or 'regression')
        raw_data: Original data before preprocessing (optional)
        processed_data: Data after preprocessing (optional)
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print("Skipping visualizations due to missing dependencies.")
        return
    
    # Create output directories
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(PREPROC_PLOT_DIR, exist_ok=True)
    
    # Visualize data before and after preprocessing if provided
    if raw_data is not None:
        print("Generating pre-processing visualizations...")
        visualize_data_distribution(raw_data, ['Range'] + feature_names, stage="raw")
        visualize_cir_patterns(raw_data, stage="raw")
        try:
            visualize_feature_correlations(raw_data, ['Range'] + feature_names, stage="raw")
        except Exception as e:
            print(f"Error generating correlation plots: {e}")
    
    if processed_data is not None:
        print("Generating post-processing visualizations...")
        visualize_data_distribution(processed_data, ['Range'] + feature_names, stage="processed", is_normalized=True)
        visualize_cir_patterns(processed_data, stage="processed")
        try:
            visualize_feature_correlations(processed_data, ['Range'] + feature_names, stage="processed")
        except Exception as e:
            print(f"Error generating correlation plots: {e}")
    
    print(f"Creating {task_type} visualization plots...")
    
    if task_type == 'classification':
        # Plot model comparison for different metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            plot_model_comparison(model_results, metric)
        
        # For each model, plot confusion matrix, feature importance, and ROC curve
        for model_name, results in model_results.items():
            print(f"\nGenerating plots for {model_name}...")
            
            # Confusion matrix
            plot_confusion_matrix(model_results, model_name)
            
            # Feature importance
            plot_feature_importance(model_results, model_name, feature_names)
            
            # ROC curve
            plot_roc_curve(model_results, model_name)
            
            # Learning curve (if sklearn is available)
            if MATPLOTLIB_AVAILABLE and NUMPY_AVAILABLE and 'model' in results:
                try:
                    from sklearn.model_selection import learning_curve
                    plot_learning_curve(X_train, y_train, model_name, results['model'])
                except ImportError:
                    pass
    
    elif task_type == 'regression':
        # Plot regression model comparison for different metrics
        for metric in ['rmse', 'mae', 'r2']:
            plot_regression_comparison(model_results, metric)
        
        # For each model, plot actual vs predicted and residuals
        for model_name, results in model_results.items():
            print(f"\nGenerating plots for {model_name}...")
            
            # Actual vs predicted
            plot_actual_vs_predicted(model_results, model_name, X_test, y_test)
            
            # Residuals
            plot_residuals(model_results, model_name, X_test, y_test)
            
            # Feature importance (if available)
            if 'feature_importance' in results and results['feature_importance'] is not None:
                plot_feature_importance(model_results, model_name, feature_names)
    
    else:
        print(f"Unknown task type: {task_type}")
        return
    
    print(f"\nAll {task_type} plots saved to {PLOT_DIR}")