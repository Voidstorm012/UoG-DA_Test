# UWB LOS/NLOS Classification and Range Prediction Models

This directory contains Jupyter notebooks for analyzing and modeling the UWB dataset for two main tasks:

## LOS/NLOS Classifier Model
The classifier model distinguishes between Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) wireless signal paths based on signal characteristics.

The notebook in `los-nlos-model/model.ipynb` includes:
- Data preparation and exploration
- Feature analysis and visualization
- Training of various classification models
- Model evaluation and comparison
- Feature importance analysis
- Recommendations based on results

## Distance Estimator Model
The distance estimator model predicts the range (distance) based on signal characteristics.

The notebook in `distance-estimator-model/model.ipynb` includes:
- Data preparation and exploration
- Feature analysis and visualization
- Training of various regression models
- Model evaluation and comparison
- Feature importance analysis
- Error analysis
- Recommendations based on results

## Setup
### Create virtual environment
```bash
python -m venv myenv
```

### Activate virtual environment
For Windows:
```bash
myenv\Scripts\activate
```
For Linux/Mac:
```bash
source myenv/bin/activate
```

### Install required packages
```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook
```bash
jupyter notebook
```

## Usage
Each notebook can be run independently to analyze and visualize each model. You can run all cells sequentially or run individual cells to explore specific parts of the analysis.

The notebooks use modules from the main project structure for data loading and preprocessing, but all model training and evaluation code is contained within the notebooks for better readability and step-by-step execution.