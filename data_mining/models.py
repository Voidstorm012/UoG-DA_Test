"""
Data Mining Module

This module handles:
1. Training different classification models
2. Evaluating model performance
3. Hyperparameter tuning
4. Feature importance analysis
"""
import sys
import random
import time

# Try to import scikit-learn and handle gracefully if not available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Using basic classifiers.")
    SKLEARN_AVAILABLE = False

# Try to import numpy and handle gracefully if not available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("Warning: NumPy not available. Using basic Python lists.")
    NUMPY_AVAILABLE = False

# Basic implementations for when scikit-learn is not available
class SimpleLogisticRegression:
    """A very simple logistic regression implementation."""
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.iterations):
            # Linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self.sigmoid(linear_model)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return y_predicted
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

class SimpleDecisionTree:
    """A very simple decision tree implementation."""
    
    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None
    
    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_classes == 1 or n_samples < 2):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, n_features)
        
        # Split the data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        # Grow the children recursively
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        
        return self.Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, n_features):
        best_feature, best_threshold = None, None
        best_gain = -float('inf')
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # Split the data
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) > 0 and np.sum(right_indices) > 0:
                    # Compute information gain
                    gain = self._information_gain(y, y[left_indices], y[right_indices])
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, parent, left_child, right_child):
        # Compute gini impurity
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        gain = self._gini(parent) - (
            weight_left * self._gini(left_child) + 
            weight_right * self._gini(right_child)
        )
        
        return gain
    
    def _gini(self, y):
        # Compute gini impurity
        m = len(y)
        if m == 0:
            return 0
        
        unique_classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        gini = 1 - np.sum(probabilities**2)
        
        return gini
    
    def _most_common_label(self, y):
        unique_classes, counts = np.unique(y, return_counts=True)
        return unique_classes[np.argmax(counts)]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)

def train_models(X_train, y_train):
    """
    Train multiple classification models.
    
    Args:
        X_train: Training features
        y_train: Training targets
    
    Returns:
        models: Dictionary of trained models
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
        print("Training Logistic Regression...")
        start_time = time.time()
        models['Logistic Regression'] = {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'scaler': scaler
        }
        models['Logistic Regression']['model'].fit(X_train_scaled, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training Random Forest...")
        start_time = time.time()
        models['Random Forest'] = {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'scaler': None  # Random Forest doesn't require scaling
        }
        models['Random Forest']['model'].fit(X_train, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training Gradient Boosting...")
        start_time = time.time()
        models['Gradient Boosting'] = {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'scaler': None  # Gradient Boosting doesn't require scaling
        }
        models['Gradient Boosting']['model'].fit(X_train, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training K-Nearest Neighbors...")
        start_time = time.time()
        models['KNN'] = {
            'model': KNeighborsClassifier(n_neighbors=5),
            'scaler': scaler
        }
        models['KNN']['model'].fit(X_train_scaled, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
        
        print("Training Support Vector Machine...")
        start_time = time.time()
        models['SVM'] = {
            'model': SVC(probability=True, random_state=42),
            'scaler': scaler
        }
        models['SVM']['model'].fit(X_train_scaled, y_train)
        print(f"  Done in {time.time() - start_time:.2f} seconds")
    else:
        # Fallback to simple implementations
        if NUMPY_AVAILABLE:
            print("Training Simple Logistic Regression...")
            start_time = time.time()
            models['Simple Logistic Regression'] = {
                'model': SimpleLogisticRegression(learning_rate=0.01, iterations=1000),
                'scaler': None
            }
            models['Simple Logistic Regression']['model'].fit(X_train, y_train)
            print(f"  Done in {time.time() - start_time:.2f} seconds")
            
            print("Training Simple Decision Tree...")
            start_time = time.time()
            models['Simple Decision Tree'] = {
                'model': SimpleDecisionTree(max_depth=5),
                'scaler': None
            }
            models['Simple Decision Tree']['model'].fit(X_train, y_train)
            print(f"  Done in {time.time() - start_time:.2f} seconds")
        else:
            print("Cannot train models: NumPy and scikit-learn are not available")
    
    return models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    
    Args:
        models: Dictionary of trained models
        X_test: Testing features
        y_test: Testing targets
    
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
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # ROC curve and AUC
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr, roc_auc = None, None, None
            
            # Feature importance (if available)
            if hasattr(model, "feature_importances_"):
                feature_importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                feature_importance = model.coef_[0]
            else:
                feature_importance = None
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'feature_importance': feature_importance
            }
            
            print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        else:
            # Simple metrics calculation
            correct = sum(1 for pred, true in zip(y_pred, y_test) if pred == true)
            accuracy = correct / len(y_test)
            
            results[name] = {
                'accuracy': accuracy
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
    
    return results