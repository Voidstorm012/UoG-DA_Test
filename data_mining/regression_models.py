"""
Regression Models Module

This module handles:
1. Training different regression models for range prediction
2. Evaluating model performance
3. Hyperparameter tuning
4. Feature importance analysis
"""
import sys
import time
import math

# Try to import scikit-learn and handle gracefully if not available
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Using basic regressors.")
    SKLEARN_AVAILABLE = False

# Try to import numpy and handle gracefully if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy not available. Using basic Python lists.")
    NUMPY_AVAILABLE = False

# Basic implementations for when scikit-learn is not available
class SimpleLinearRegression:
    """A very simple linear regression implementation."""
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            # Linear model
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    @property
    def coef_(self):
        return self.weights

def train_regression_models(X_train, y_train):
    """
    Train multiple regression models for range prediction.
    
    Args:
        X_train: Training features
        y_train: Training targets (range values)
    
    Returns:
        models: Dictionary of trained regression models
    """
    models = {}
    
    # Convert to numpy arrays if not already
    if NUMPY_AVAILABLE and not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
    
    # Standardize features
    if SKLEARN_AVAILABLE:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train models
        print("Training Linear Regression...")
        start_time = time.time()
        models['Linear Regression'] = {
            'model': LinearRegression(),
            'scaler': scaler
        }
        models['Linear Regression']['model'].fit(X_train_scaled, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training Ridge Regression...")
        start_time = time.time()
        models['Ridge Regression'] = {
            'model': Ridge(alpha=1.0),
            'scaler': scaler
        }
        models['Ridge Regression']['model'].fit(X_train_scaled, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training Random Forest Regressor...")
        start_time = time.time()
        models['Random Forest'] = {
            'model': RandomForestRegressor(n_estimators=100, random_state=42),
            'scaler': None  # Random Forest doesn't require scaling
        }
        models['Random Forest']['model'].fit(X_train, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training Gradient Boosting Regressor...")
        start_time = time.time()
        models['Gradient Boosting'] = {
            'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'scaler': None  # Gradient Boosting doesn't require scaling
        }
        models['Gradient Boosting']['model'].fit(X_train, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training SVR...")
        start_time = time.time()
        models['SVR'] = {
            'model': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'scaler': scaler
        }
        models['SVR']['model'].fit(X_train_scaled, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
    else:
        # Fallback to simple implementations
        if NUMPY_AVAILABLE:
            print("Training Simple Linear Regression...")
            start_time = time.time()
            models['Simple Linear Regression'] = {
                'model': SimpleLinearRegression(learning_rate=0.01, iterations=1000),
                'scaler': None
            }
            models['Simple Linear Regression']['model'].fit(X_train, y_train)
            print(f"  Done in {time.time() - start_time:.2f} seconds")
        else:
            print("Cannot train models: NumPy and scikit-learn are not available")
    
    return models

def evaluate_regression_models(models, X_test, y_test):
    """
    Evaluate trained regression models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Testing features
        y_test: Testing targets (range values)
    
    Returns:
        results: Dictionary of evaluation metrics for each model
    """
    results = {}
    
    # Convert to numpy arrays if not already
    if NUMPY_AVAILABLE and not isinstance(X_test, np.ndarray):
        X_test = np.array(X_test)
        y_test = np.array(y_test)
    
    for name, model_info in models.items():
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale the test data if needed
        X_test_scaled = X_test
        if scaler is not None:
            X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        if SKLEARN_AVAILABLE and NUMPY_AVAILABLE:
            mse = mean_squared_error(y_test, y_pred)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance (if available)
            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                feature_importance = np.abs(model.coef_)
            else:
                feature_importance = None
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'feature_importance': feature_importance
            }
            
            print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        else:
            # Simple metrics calculation
            if NUMPY_AVAILABLE:
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = math.sqrt(mse)
                mae = np.mean(np.abs(y_test - y_pred))
                
                results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae
                }
                
                print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            else:
                print("Cannot calculate metrics: NumPy is not available")
    
    return results