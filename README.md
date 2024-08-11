# Project Title

**Air quality index prediction (IIT kharagpur hackathon)**

## Overview

This project is a machine learning solution that involves the use of various classification models to analyze and predict outcomes based on the provided dataset. The code demonstrates how to preprocess the data, train multiple machine learning models, and evaluate their performance using different metrics.

## Project Structure

- **Data Preprocessing**: The notebook includes steps to load the dataset, clean it, and prepare it for training the machine learning models.
- **Model Training**: Several models are trained on the dataset, including:
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Decision Tree Classifier
  - Logistic Regression
  - Random Forest Classifier
  - Naive Bayes
  - XGBoost
  - LightGBM
- **Model Evaluation**: The models are evaluated using metrics such as accuracy, confusion matrix, and classification report.

## Requirements

The following Python libraries are required to run the notebook:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`
- `lightgbm`

If you do not have these installed, the notebook includes a cell to install `xgboost` and `lightgbm` using pip.

## Installation

You can install the required libraries using the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
```

## How to Run the Notebook

1. **Load the Dataset**: The dataset is loaded from a CSV file. Ensure you have the dataset in the correct path or update the path in the code accordingly.
2. **Run the Cells**: Execute the cells in sequence to preprocess the data, train the models, and evaluate their performance.
3. **Model Evaluation**: After training, the notebook will output various performance metrics and visualizations to help assess each model.

## Dataset

The dataset used in this project is referred to in the notebook as `Train_Data_Final.csv`. Make sure this dataset is accessible to the notebook by placing it in the specified path or updating the code to reflect its location.

## Model Accuracies

| Model Name                 | Accuracy (%) |
|----------------------------|--------------|
| K-Nearest Neighbors (KNN)   |   53.1%           |
| Support Vector Classifier   |   64.8%           |
| Decision Tree Classifier    |   94.6%           |
| Random Forest Classifier    |   81.4%           |
| Naive Bayes                 |   39.8%           |
| XGBoost                     |   95.4%           |
| LightGBM                    |   95.7%           |
