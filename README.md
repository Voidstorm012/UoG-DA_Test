# UWB LOS/NLOS Classification and Range Prediction Project (Group 17)

This project implements machine learning techniques for two key tasks in UWB-based indoor positioning:

1. **LOS/NLOS Classification**: Identify whether a signal path is Line-of-Sight (LOS) or Non-Line-of-Sight (NLOS) based on signal characteristics.
2. **Range Prediction**: Estimate the measured range based on signal parameters, allowing for more accurate distance calculations.

Accurate classification and range prediction help improve indoor positioning accuracy by identifying and properly handling NLOS measurements.

## Project Structure

```
UoG-DA_Group01/
├── data/                   # Dataset files and loading utilities
│   ├── code/               # Dataset import module
│   └── dataset/            # CSV data files (7 parts)
├── data_preparation/       # Data loading, cleaning, preprocessing
├── data_mining/            # Machine learning models and evaluation
├── data_visualization/     # Plotting and visualization tools
├── notebooks/              # Jupyter notebooks for interactive analysis
│   ├── los-nlos-model/     # LOS/NLOS classification notebook
│   └── distance-estimator-model/ # Range prediction notebook
├── results/                # Output reports and analysis
│   ├── plots/              # Visualization plots
│   └── reports/            # Analysis reports
├── main.py                 # Main entry point (full pipeline)
└── README.md               # Project documentation
```

## Dataset

The project uses the UWB LOS/NLOS dataset, which contains:
- 42,000 samples (21,000 LOS, 21,000 NLOS)
- Data collected from 7 different indoor environments (office, apartment, workshop, kitchen, bedroom, etc.)
- 15 feature columns + 1016 Channel Impulse Response (CIR) samples per measurement

### Key Features
- **NLOS Flag**: Binary classification target (0=LOS, 1=NLOS)
- **Range**: Measured distance (regression target)
- **Signal Parameters**: FP_IDX, FP_AMP1/2/3, STDEV_NOISE, CIR_PWR, etc.
- **Configuration Parameters**: CH, FRAME_LEN, PREAM_LEN, BITRATE, PRFR
- **CIR Values**: 1016 samples of Channel Impulse Response

## Getting Started

### Prerequisites

- Python 3.6+
- Required packages: numpy, pandas, matplotlib, scikit-learn, jupyter (for notebooks)
- Optional packages: seaborn (enhanced visualizations)

### Running Options

#### Option 1: Using Jupyter Notebooks (Recommended for Interactive Analysis)

```bash
# Navigate to the notebooks directory
cd UoG-DA_Test/notebooks

# Install required packages
pip install -r requirements.txt

# Start Jupyter Notebook
jupyter notebook
```

Then open either:
- `los-nlos-model/model.ipynb` for LOS/NLOS classification
- `distance-estimator-model/model.ipynb` for range prediction

The notebooks provide an interactive environment where you can run code blocks individually and see the results immediately, with detailed visualizations and analysis.

#### Option 2: Using Command Line (For Full Pipeline Execution)

```bash
# Navigate to the project directory
cd UoG-DA_Test

# Install required packages
pip install numpy pandas matplotlib scikit-learn

# Run both classification and range prediction pipelines (default)
python main.py

# Run only the classification pipeline
python main.py --task classification

# Run only the range prediction pipeline
python main.py --task regression
```

This approach runs the complete pipeline from data preparation to results analysis in a single execution.

## Analysis Pipeline

The project follows a standard data analytics pipeline:

1. **Data Preparation**
   - Loading the UWB dataset from CSV files
   - Normalizing CIR values by RX preamble count (RXPACC)
   - Feature selection and engineering
   - Train-test splitting (80/20)

2. **Data Mining**
   - **Classification Task**:
     - Training multiple classification models (Logistic Regression, Random Forest, etc.)
     - Model evaluation and comparison using accuracy, precision, recall, F1-score
     - Feature importance analysis
   - **Regression Task**:
     - Training regression models for range prediction
     - Evaluation with RMSE, MAE, and R² metrics
     - Feature importance for range prediction

3. **Data Visualization**
   - Class distribution visualization
   - Feature distributions by class (LOS/NLOS)
   - Performance metrics comparison
   - Confusion matrices and ROC curves
   - Actual vs. predicted plots for regression
   - Feature importance visualizations

4. **Results Analysis**
   - Model performance interpretation
   - Theoretical analysis for both classification and regression tasks
   - Recommendations for deployment in real-world scenarios

## Key Features of This Implementation

- **Two Execution Options**: Run either through interactive Jupyter notebooks or as a complete pipeline
- **Comprehensive Visualizations**: Detailed plots for better understanding of data and results
- **Multiple Model Comparison**: Train and evaluate several models to find the best performer
- **Feature Importance Analysis**: Identify which signal characteristics are most important
- **Modular Design**: Well-organized code structure for easy maintenance and extension

The project generates comprehensive analyses for both tasks:

### LOS/NLOS Classification Results
- Comparison of different classification model performances
- Identification of the most important features for LOS/NLOS detection
- Feature distributions before and after preprocessing
- Theoretical explanation of classification results
- Practical recommendations for LOS/NLOS classification implementation

### Range Prediction Results
- Comparison of regression model performances (RMSE, MAE, R²)
- Analysis of key features influencing range prediction
- Theoretical understanding of range estimation in LOS vs. NLOS conditions
- Recommendations for improving range prediction accuracy

## Key Enhancements in This Version

- **Enhanced Data Visualization**: Added pre- and post-processing visualizations to better understand data characteristics
- **CIR Pattern Analysis**: Visualize and compare CIR patterns between LOS and NLOS signals
- **Feature Correlation Analysis**: Heatmaps showing relationships between different signal features
- **Robust Error Handling**: Better handling of package dependencies and error conditions
- **Improved Documentation**: More comprehensive README and code comments

## License

This project is licensed under the CC BY 4.0 License - see the LICENSE.txt file for details.

## Acknowledgments

- Based on the UWB LOS/NLOS dataset by Klemen Bregar, Andrej Hrovat, and Mihael Mohorčič.
- Original dataset: [UWB-LOS-NLOS-Data-Set](https://github.com/ewine-project/UWB-LOS-NLOS-Data-Set)
