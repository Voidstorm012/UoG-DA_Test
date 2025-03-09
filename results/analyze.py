"""
Results Analysis Module

This module handles:
1. Interpretation of model results
2. Summarizing model performance
3. Theoretical explanations
4. Recommendations for model selection
"""
import os
import time

# Create output directory for analysis reports
REPORT_DIR = os.path.join('..', 'results', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def find_best_model(model_results, metric='accuracy', task_type='classification'):
    """
    Find the best model based on a specific metric.
    
    Args:
        model_results: Dictionary of model evaluation results
        metric: Metric to compare (accuracy, precision, recall, f1_score for classification;
                rmse, mae, r2 for regression)
        task_type: Type of task ('classification' or 'regression')
    
    Returns:
        best_model: Name of the best model
        best_value: Value of the metric for the best model
    """
    best_model = None
    
    if task_type == 'classification':
        # For classification metrics, higher is better
        best_value = -1
        for model_name, results in model_results.items():
            if metric in results and results[metric] > best_value:
                best_value = results[metric]
                best_model = model_name
    
    elif task_type == 'regression':
        if metric == 'r2':
            # For R², higher is better
            best_value = -float('inf')
            for model_name, results in model_results.items():
                if metric in results and results[metric] > best_value:
                    best_value = results[metric]
                    best_model = model_name
        else:
            # For RMSE and MAE, lower is better
            best_value = float('inf')
            for model_name, results in model_results.items():
                if metric in results and results[metric] < best_value:
                    best_value = results[metric]
                    best_model = model_name
    
    return best_model, best_value

def rank_features(model_results, feature_names):
    """
    Rank features by importance across all models.
    
    Args:
        model_results: Dictionary of model evaluation results
        feature_names: List of feature names
    
    Returns:
        ranked_features: Dictionary with feature rankings
    """
    feature_scores = {feature: 0 for feature in feature_names}
    count = 0
    
    for _, results in model_results.items():
        if 'feature_importance' in results and results['feature_importance'] is not None:
            count += 1
            importance = results['feature_importance']
            
            # Normalize importance scores
            total = sum(abs(imp) for imp in importance)
            if total == 0:
                continue
            
            normalized_importance = [abs(imp) / total for imp in importance]
            
            # Add to feature scores
            for i, feature in enumerate(feature_names):
                feature_scores[feature] += normalized_importance[i]
    
    # Average scores
    if count > 0:
        for feature in feature_scores:
            feature_scores[feature] /= count
    
    # Rank features
    ranked_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_features

def generate_summary_report(model_results, feature_rankings, task_type='classification'):
    """
    Generate a summary report of model results.
    
    Args:
        model_results: Dictionary of model evaluation results
        feature_rankings: Ranked features by importance
        task_type: Type of task ('classification' or 'regression')
    """
    report_path = os.path.join(REPORT_DIR, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        if task_type == 'classification':
            f.write("UWB LOS/NLOS Classification - Summary Report\n")
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        else:  # regression
            f.write("UWB Range Prediction - Summary Report\n")
            metrics = ['rmse', 'mae', 'r2']
        
        f.write("=" * 50 + "\n\n")
        
        # Model performance summary
        f.write("Model Performance Summary\n")
        f.write("-" * 30 + "\n\n")
        
        for metric in metrics:
            best_model, best_value = find_best_model(model_results, metric, task_type)
            if best_model:
                f.write(f"Best model for {metric.upper() if metric != 'r2' else 'R²'}: {best_model} ({best_value:.4f})\n")
        
        f.write("\n")
        
        # Feature importance summary
        f.write("Feature Importance Ranking\n")
        f.write("-" * 30 + "\n\n")
        
        for i, (feature, score) in enumerate(feature_rankings):
            f.write(f"{i+1}. {feature}: {score:.4f}\n")
        
        f.write("\n")
        
        # Model details
        f.write("Individual Model Performance\n")
        f.write("-" * 30 + "\n\n")
        
        for model_name, results in model_results.items():
            f.write(f"Model: {model_name}\n")
            
            for metric in metrics:
                if metric in results:
                    metric_name = metric.upper() if metric != 'r2' else 'R²'
                    f.write(f"  {metric_name}: {results[metric]:.4f}\n")
            
            f.write("\n")
    
    print(f"Summary report saved to {report_path}")
    return report_path

def write_theoretical_analysis(ranked_features, task_type='classification'):
    """
    Write a theoretical analysis of the results.
    
    Args:
        ranked_features: Ranked features by importance
        task_type: Type of task ('classification' or 'regression')
    """
    analysis_path = os.path.join(REPORT_DIR, 'theoretical_analysis.txt')
    
    uwb_theory = {
        'NLOS': 'The NLOS flag indicates whether there is a direct line of sight between devices. This directly affects the range estimation as NLOS conditions introduce additional signal travel distance.',
        
        'Range': 'The measured range (time of flight) is a direct indicator of distance between devices. In NLOS scenarios, the signal must travel through or around obstacles, typically resulting in a longer measured range compared to the true distance.',
        
        'FP_IDX': 'The first path index indicates where in the CIR the first path was detected. In NLOS scenarios, multipath effects may cause the strongest path to not be the first path, affecting this index.',
        
        'FP_AMP1': 'First path amplitude (part 1) is typically stronger in LOS conditions as the signal has not been attenuated by obstacles.',
        
        'FP_AMP2': 'First path amplitude (part 2) follows similar patterns to FP_AMP1, providing additional signal strength information.',
        
        'FP_AMP3': 'First path amplitude (part 3) completes the amplitude characterization, with all three parts together giving a complete picture of signal strength.',
        
        'STDEV_NOISE': 'Standard deviation of noise tends to be higher in NLOS scenarios due to increased multipath components and signal scattering.',
        
        'CIR_PWR': 'Total channel impulse response power is often lower in NLOS conditions due to signal attenuation through obstacles.',
        
        'MAX_NOISE': 'Maximum noise value can indicate the presence of strong reflections or interference, which are more common in NLOS scenarios.',
        
        'RXPACC': 'Received preamble count affects the reliability of measurements, with higher counts generally providing more reliable readings.',
        
        'CH': 'The UWB channel used can affect performance as different frequencies have different propagation characteristics through obstacles.',
        
        'FRAME_LEN': 'Frame length is a technical parameter that may indirectly correlate with NLOS conditions based on the device configuration.',
        
        'PREAM_LEN': 'Preamble length affects the ability to synchronize and detect the signal accurately, especially in challenging NLOS environments.',
        
        'BITRATE': 'Data rate can correlate with modulation schemes that have different robustness to NLOS conditions.',
        
        'PRFR': 'Pulse repetition frequency affects the density of measurements and can impact the ability to detect signals in NLOS conditions.'
    }
    
    with open(analysis_path, 'w') as f:
        if task_type == 'classification':
            f.write("UWB LOS/NLOS Classification - Theoretical Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Introduction\n")
            f.write("-" * 20 + "\n\n")
            f.write("Ultra-Wideband (UWB) technology operates in the 3.1-10 GHz frequency range and provides high-resolution positioning data. One of the key challenges in indoor positioning using UWB is distinguishing between Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) conditions, as NLOS conditions can significantly degrade positioning accuracy.\n\n")
            
            f.write("Feature Importance Analysis\n")
            f.write("-" * 20 + "\n\n")
            f.write("The machine learning models have identified the following features as most important for distinguishing between LOS and NLOS conditions:\n\n")
            
            for i, (feature, score) in enumerate(ranked_features[:5]):
                f.write(f"{i+1}. {feature} (Importance: {score:.4f})\n")
                if feature in uwb_theory:
                    f.write(f"   Theoretical relevance: {uwb_theory[feature]}\n\n")
            
            f.write("\nPhysical Interpretation of Results\n")
            f.write("-" * 20 + "\n\n")
            f.write("In indoor environments, UWB signals interact with the physical environment in complex ways. In LOS conditions, signals travel directly from transmitter to receiver with minimal distortion. In NLOS conditions, signals encounter obstacles such as walls, furniture, and people, causing:\n\n")
            f.write("1. Signal attenuation (reduction in strength)\n")
            f.write("2. Multipath propagation (multiple signal paths from reflections)\n")
            f.write("3. Diffraction (signals bending around obstacles)\n")
            f.write("4. Scattering (signals dispersing in multiple directions)\n\n")
            
            f.write("These physical phenomena manifest in the UWB channel impulse response (CIR) and the features derived from it. The machine learning models are able to detect patterns in these features that reliably distinguish between LOS and NLOS conditions.\n\n")
            
            f.write("Practical Implications\n")
            f.write("-" * 20 + "\n\n")
            f.write("Accurate LOS/NLOS classification has several practical implications for indoor positioning systems:\n\n")
            f.write("1. Improved position accuracy by identifying and compensating for NLOS conditions\n")
            f.write("2. Reduced deployment costs by requiring fewer anchors for reliable positioning\n")
            f.write("3. Enhanced robustness in dynamic environments with changing obstacle configurations\n")
            f.write("4. Better user experience in applications like augmented reality, asset tracking, and robotics\n\n")
            
            f.write("Future Research Directions\n")
            f.write("-" * 20 + "\n\n")
            f.write("Based on our findings, future research could explore:\n\n")
            f.write("1. Incorporating temporal patterns in UWB measurements to improve classification\n")
            f.write("2. Combining UWB with other sensing modalities (e.g., IMU, camera) for better NLOS identification\n")
            f.write("3. Developing efficient online learning approaches that can adapt to changing environments\n")
            f.write("4. Exploring advanced CIR feature extraction techniques using deep learning\n")
        
        else:  # regression
            f.write("UWB Range Prediction - Theoretical Analysis\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Introduction\n")
            f.write("-" * 20 + "\n\n")
            f.write("Ultra-Wideband (UWB) technology provides high-precision range measurements by using time-of-flight calculations between devices. Accurate range estimation is crucial for precise indoor localization, but the measured range can be affected by various environmental factors and signal characteristics.\n\n")
            
            f.write("Feature Importance Analysis\n")
            f.write("-" * 20 + "\n\n")
            f.write("The regression models have identified the following features as most important for predicting accurate range measurements:\n\n")
            
            for i, (feature, score) in enumerate(ranked_features[:5]):
                f.write(f"{i+1}. {feature} (Importance: {score:.4f})\n")
                if feature in uwb_theory:
                    f.write(f"   Theoretical relevance: {uwb_theory[feature]}\n\n")
            
            f.write("\nPhysical Interpretation of Results\n")
            f.write("-" * 20 + "\n\n")
            f.write("Range measurements in UWB systems are based on the time of flight of radio signals. However, several physical phenomena affect these measurements:\n\n")
            f.write("1. NLOS conditions add extra distance as signals must travel around or through obstacles\n")
            f.write("2. Multipath propagation creates multiple reflections that can cause range estimation errors\n")
            f.write("3. Signal strength variations affect the detection of the first arriving path\n")
            f.write("4. Environmental factors like humidity, temperature, and material properties influence signal propagation\n\n")
            
            f.write("The regression models learn to compensate for these factors by identifying patterns in how they affect the measured range.\n\n")
            
            f.write("Practical Implications\n")
            f.write("-" * 20 + "\n\n")
            f.write("Accurate range prediction, even in challenging NLOS conditions, has significant practical implications:\n\n")
            f.write("1. Improved localization accuracy in indoor environments with obstacles\n")
            f.write("2. Reduced need for dense anchor deployment by better handling of NLOS measurements\n")
            f.write("3. Enhanced tracking consistency in dynamic environments\n")
            f.write("4. More reliable positioning for applications like robotics, healthcare monitoring, and industrial automation\n\n")
            
            f.write("Future Research Directions\n")
            f.write("-" * 20 + "\n\n")
            f.write("Based on our findings, promising future research directions include:\n\n")
            f.write("1. Hybrid models that combine classification (LOS/NLOS) and regression (range correction)\n")
            f.write("2. Deep learning approaches that can extract more complex patterns from the CIR\n")
            f.write("3. Transfer learning methods to adapt models to new environments quickly\n")
            f.write("4. Integration with complementary technologies like IMU and vision for robust localization\n")
    
    print(f"Theoretical analysis saved to {analysis_path}")
    return analysis_path

def generate_recommendations(task_type='classification'):
    """
    Generate recommendations based on the analysis.
    
    Args:
        task_type: Type of task ('classification' or 'regression')
    """
    recommendations_path = os.path.join(REPORT_DIR, 'recommendations.txt')
    
    with open(recommendations_path, 'w') as f:
        if task_type == 'classification':
            f.write("UWB LOS/NLOS Classification - Recommendations\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Selection\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. For optimal accuracy, ensemble methods like Random Forest and Gradient Boosting generally perform best on this dataset due to their ability to capture complex relationships between features.\n\n")
            f.write("2. If computational resources are limited (e.g., in embedded systems), simpler models like Decision Trees or Logistic Regression can provide a good balance between performance and efficiency.\n\n")
            f.write("3. For deployment in real-time systems, consider using models that offer faster inference times, such as optimized implementations of Decision Trees.\n\n")
            
            f.write("Feature Engineering\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. Focus on the top-ranked features identified in our analysis, particularly those related to signal amplitude and power characteristics.\n\n")
            f.write("2. Consider engineered features that combine multiple raw measurements, such as ratios between different amplitude components.\n\n")
            f.write("3. For improved performance, explore frequency-domain features derived from the Channel Impulse Response (CIR).\n\n")
            
            f.write("Deployment Strategies\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. Implement a two-stage positioning approach where LOS/NLOS classification is performed first, followed by position calculation using only LOS measurements.\n\n")
            f.write("2. In environments with frequent NLOS conditions, consider deploying additional anchors to ensure that enough LOS measurements are available.\n\n")
            f.write("3. Periodically retrain the model using data collected from the specific deployment environment to account for environmental changes.\n\n")
            
            f.write("Performance Monitoring\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. Implement a confidence score based on the model's probability output to flag uncertain classifications.\n\n")
            f.write("2. Regularly validate the model's performance using ground truth data collected in the deployment environment.\n\n")
            f.write("3. Consider ensemble approaches that combine multiple models for more robust classification, especially in challenging environments.\n")
        
        else:  # regression
            f.write("UWB Range Prediction - Recommendations\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model Selection\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. For optimal accuracy in range prediction, ensemble methods like Random Forest and Gradient Boosting typically offer the best performance by capturing non-linear relationships between features and range values.\n\n")
            f.write("2. For scenarios requiring low computational resources, consider using Ridge Regression which can provide a good balance between prediction accuracy and efficiency.\n\n")
            f.write("3. When the relationship between certain features and range is particularly non-linear, SVR with appropriate kernel functions can model these relationships effectively.\n\n")
            
            f.write("Feature Engineering\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. The NLOS flag is a critical feature for range prediction - consider creating separate models for LOS and NLOS scenarios for specialized accuracy.\n\n")
            f.write("2. Develop compound features that combine signal strength parameters with noise characteristics to capture interaction effects.\n\n")
            f.write("3. Extract additional features from the raw CIR data, such as peak counts, power distribution metrics, and delay spread measurements.\n\n")
            
            f.write("Deployment Strategies\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. Implement a cascaded approach: first classify LOS/NLOS condition, then apply the appropriate range prediction model based on this classification.\n\n")
            f.write("2. For critical applications, use different range prediction models for different environments (e.g., open spaces vs. crowded areas).\n\n")
            f.write("3. Consider using a weighted approach where multiple range prediction models contribute to the final range estimate.\n\n")
            
            f.write("Performance Optimization\n")
            f.write("-" * 20 + "\n\n")
            f.write("1. Regularly update models with new environmental data to maintain accuracy as conditions change.\n\n")
            f.write("2. Implement confidence intervals for range predictions to identify measurements with high uncertainty.\n\n")
            f.write("3. For applications requiring extremely high accuracy, integrate the range prediction with complementary sensors like IMU or camera systems to improve overall performance.\n")
    
    print(f"Recommendations saved to {recommendations_path}")
    return recommendations_path

def interpret_results(model_results, feature_names, task_type='classification'):
    """
    Interpret model results and generate analysis reports.
    
    Args:
        model_results: Dictionary of model evaluation results
        feature_names: List of feature names
        task_type: Type of task ('classification' or 'regression')
    """
    print(f"Analyzing {task_type} model results...")
    
    # Create output directory
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Rank features by importance
    feature_rankings = rank_features(model_results, feature_names)
    
    # Generate summary report
    summary_path = generate_summary_report(model_results, feature_rankings, task_type)
    
    # Write theoretical analysis
    analysis_path = write_theoretical_analysis(feature_rankings, task_type)
    
    # Generate recommendations
    recommendations_path = generate_recommendations(task_type)
    
    print(f"\nAnalysis completed. Reports saved to {REPORT_DIR}")
    print(f"  - Summary report: {summary_path}")
    print(f"  - Theoretical analysis: {analysis_path}")
    print(f"  - Recommendations: {recommendations_path}")
    
    return {
        'summary_path': summary_path,
        'analysis_path': analysis_path,
        'recommendations_path': recommendations_path
    }