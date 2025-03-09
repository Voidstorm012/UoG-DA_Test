# UWB LOS/NLOS Classification Project

**Group 01 | Data Analytics Course | University of Glasgow | March 2025**

## Team Members and Contributions

*[List of team members with their roles and contributions]*

## 1. Problem Definition

### 1.1 Background

Indoor positioning systems require accurate distance measurements between anchors and tags. In Ultra-Wideband (UWB) systems, Non-Line-of-Sight (NLOS) conditions significantly degrade ranging accuracy, leading to positioning errors. This project focuses on solving two critical challenges:

1. **Classification Challenge**: Accurately identifying whether a UWB measurement is from a Line-of-Sight (LOS) or Non-Line-of-Sight (NLOS) condition.
2. **Range Estimation Challenge**: Predicting the true distance between devices, especially in NLOS conditions where measured ranges are typically biased.

### 1.2 Importance and Applications

Accurate LOS/NLOS classification and range estimation are essential for:

- **Indoor Positioning Systems**: Enabling centimeter-level accuracy in indoor environments
- **Robotics and Autonomous Navigation**: Providing reliable positioning in complex indoor settings
- **Asset Tracking**: Improving inventory management and logistics operations
- **Smart Buildings**: Enhancing occupancy detection and room-level positioning
- **AR/VR Applications**: Supporting precise spatial mapping and user positioning

### 1.3 Project Objectives

This project aims to:

1. Develop an accurate classifier to distinguish between LOS and NLOS UWB measurements
2. Identify the most significant features that affect LOS/NLOS conditions
3. Create a range estimation model to correct for biases in NLOS measurements
4. Evaluate model performance using rigorous metrics and cross-validation

## 2. Problem Analysis

### 2.1 Dataset Overview

The UWB LOS/NLOS dataset contains:
- 42,000 samples (21,000 LOS, 21,000 NLOS)
- Data collected from 7 different indoor environments
- 15 metadata features + 1016 Channel Impulse Response (CIR) values per sample

### 2.2 Initial Data Exploration

Our exploratory analysis revealed:
- A balanced dataset (50% LOS, 50% NLOS)
- Significant differences in signal characteristics between LOS and NLOS conditions
- Strong correlations between certain features and LOS/NLOS classification
- Variations in CIR patterns indicative of multipath effects in NLOS scenarios

### 2.3 Key Challenges

The main challenges identified include:
- High dimensionality due to the 1016 CIR values
- Potential redundancy in features
- Variations across different indoor environments
- Need for robust models that generalize to new environments

### 2.4 Novelty of Our Approach

Our solution is novel in several ways:
- We apply feature engineering to extract key characteristics from the CIR
- We implement a two-stage pipeline: classification followed by range estimation
- We utilize ensemble learning to improve robustness and accuracy
- We incorporate domain knowledge about UWB signal propagation into our models

## 3. Methodology

### 3.1 Data Preparation and Preprocessing

#### 3.1.1 Data Cleaning
- Identification and handling of outliers
- Normalization of CIR values by RXPACC (received preamble count)
- Feature scaling to ensure algorithm convergence

#### 3.1.2 Feature Selection and Engineering
- Statistical analysis of feature importance
- Extraction of key CIR characteristics (peak amplitude, rise time, etc.)
- Creation of new engineered features (ratios, statistical moments, etc.)

#### 3.1.3 Dimensionality Reduction
- PCA analysis to reduce CIR dimensionality
- Selection of top features based on their importance scores

### 3.2 Data Mining Algorithms

#### 3.2.1 Classification Models
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine
- Neural Network
- K-Nearest Neighbors

#### 3.2.2 Range Estimation Models
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression with polynomial features

#### 3.2.3 Hyperparameter Tuning
- Grid search for optimal hyperparameters
- Cross-validation to prevent overfitting

### 3.3 Evaluation Methodology

#### 3.3.1 Classification Metrics
- Accuracy, Precision, Recall, F1-score
- ROC curve and AUC
- Confusion matrix

#### 3.3.2 Range Estimation Metrics
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² score

## 4. Results and Analysis

### 4.1 Classification Results

#### 4.1.1 Model Performance Comparison
*[Results table comparing the performance of different classification models]*

#### 4.1.2 Feature Importance Analysis
*[Analysis of the most important features for classification]*

#### 4.1.3 ROC Curves
*[ROC curves for different classification models]*

### 4.2 Range Estimation Results

#### 4.2.1 Model Performance Comparison
*[Results table comparing the performance of different regression models]*

#### 4.2.2 Error Distribution Analysis
*[Analysis of error distributions in range estimation]*

### 4.3 Detailed Analysis

#### 4.3.1 Environment-Specific Performance
*[Analysis of model performance across different environments]*

#### 4.3.2 Feature Correlation Analysis
*[Correlation analysis between features and target variables]*

## 5. Discussion

### 5.1 Interpretation of Results

Our results show that:
- Feature X, Y, and Z are the most significant indicators of NLOS conditions
- Ensemble methods (Random Forest, Gradient Boosting) outperform other algorithms
- CIR characteristics provide valuable information beyond metadata features
- Range estimation accuracy varies with distance and environmental complexity

### 5.2 Practical Implications

The practical implications of our findings include:
- Recommendations for UWB system deployment in complex environments
- Guidelines for anchor placement to minimize NLOS conditions
- Strategies for integrating our models into real-time positioning systems
- Suggested enhancements to UWB hardware and signal processing

### 5.3 Limitations

Our approach has the following limitations:
- Dependence on the environments represented in the training data
- Computational complexity of processing full CIR data
- Challenges in real-time implementation on resource-constrained devices

## 6. Conclusion and Future Work

### 6.1 Conclusion

We successfully developed and evaluated models for UWB LOS/NLOS classification and range estimation. Our approaches achieved high accuracy in distinguishing between LOS and NLOS conditions and provided significant improvements in range estimation for NLOS scenarios.

### 6.2 Future Work

Potential directions for future research include:
- Integration with sensor fusion approaches (IMU, camera, etc.)
- Online learning algorithms that adapt to new environments
- Deep learning architectures for end-to-end CIR processing
- Transfer learning to adapt models to new UWB hardware

## 7. References

1. *[List of references]*

## Appendix

### A. Algorithm Pseudocode

*[Pseudocode for key algorithms]*

### B. Additional Visualizations

*[Additional plots and visualizations]*

### C. Implementation Details

*[Details on implementation and software used]*