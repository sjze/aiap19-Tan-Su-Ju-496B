# Machine Learning Pipeline

## Overview
This repository contains an end-to-end machine learning pipeline for predicting temperature conditions and categorizing plant types and stages based on sensor data from AgroTech Innovations.

## Folder Structure
- `src/`: Contains Python modules for data loading, preprocessing, model training, and evaluation.
- `run.sh`: A bash script to execute the pipeline.
- `requirements.txt`: Required Python packages.
- `README.md`: Overview of the pipeline and instructions.

## Requirements
To set up the environment:
```bash
python -m venv myenv
source myenv/bin/activate 
pip install -r requirements.txt
```

## File structure
├── src
│   ├── data_loader.py          # Data loading and SQL query handling
│   ├── data_preprocessor.py    # Data preprocessing and feature engineering
│   ├── model_trainer.py        # Model training and optimization
│   ├── model_evaluator.py      # Model evaluation and metrics calculation
│   ├── main.py                 # Main entry point for running the pipeline
├── data
│   └── agri.db                 # The dataset (SQLite database)
├── .gitignore                  # Git ignore configuration
├── README.md                   # This file
├── requirements.txt            # List of required Python dependencies
└── run.sh                      # Bash script to run the pipeline


## Instructions to run for regression. Task1: Prediction of of temperature conditions
- Choices for model=['regression', 'classification']
- Choices for task = ['linear_regression', 'random_forest', 'xgboost']
```bash 
./run.sh --task regression --model random_forest  
```

## Instructions to run for classification Task2: Categorise combined PLANT-TYPE STAGE
```bash
./run.sh --task classification --model xgboost  
```


The submitted MLP, including the `README.md`, will be used to assess your understanding of machine learning models/algorithms as well as your ability to design and develop a machine learning pipeline. Specifically, you will be assessed on
1. Appropriate data preprocessing and feature engineering
2. Appropriate use and optimization of algorithms/models
3. Appropriate explanation for the choice of algorithms/models
4. Appropriate use of evaluation metrics
5. Appropriate explanation for the choice of evaluation metrics
6. Understanding of the different components in the machine learning pipeline

## Data preprocessingand Feature Engineering
- Target Column Identification: Depending on the task type (regression or classification), the appropriate target column is selected. For regression, the target is temperature (Temperature Sensor (°C)), while for classification, it's a combined label (Plant Type-Stage).

- Imputation and Encoding: Missing data is handled by imputation (mean or mode depending on task). Categorical features are one-hot encoded using OneHotEncoder, and numeric features are standardized or scaled as needed.

- Feature Selection: The features related to the system location code and previous plant type (which are not relevant for model prediction) are dropped to avoid noise.

## Choice of models
- Random Forest and XGBoost are chosen for classification and regression tasks due to their robustness and ability to handle large datasets with minimal tuning. These models are known for their high accuracy and interpretability, particularly in predicting sensor readings or classifying plant stages.

- Linear regression and logistic regression models are chosen to test for comparison on scores. Tested as a baseline for the regression task but was outperformed by more complex models like Random Forest and XGBoost.

Random Forest: Selected for its ability to capture complex relationships between features without overfitting.
XGBoost: Chosen for its performance in handling large datasets with boosting methods to reduce bias and variance.

## Evaluation Metrics
- For regression tasks: Metrics like Mean Squared Error (MSE) and R² Score are used to assess model performance. These metrics measure the model's prediction error and how well the model fits the data.

- For classification tasks: Metrics such as Accuracy, Precision, Recall, and F1-Score are used to evaluate classification performance. These metrics provide a balanced view of how well the model identifies plant stages.

