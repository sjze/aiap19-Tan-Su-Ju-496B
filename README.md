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

## Instructions to run
# Choices for model=['regression', 'classification']
# Choices for task = ['linear_regression', 'random_forest', 'xgboost']
```bash 
./run.sh --task regression --model random_forest  

./run.sh --task classification --model xgboost  
```