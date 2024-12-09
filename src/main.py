import argparse
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
import numpy as np


# Running the pipelne
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Machine Learning Pipeline')
    parser.add_argument('--task', type=str, default='regression', choices=['regression', 'classification'], help='Choose task type.')
    parser.add_argument('--model', type=str, default='random_forest', choices=['linear_regression', 'random_forest', 'xgboost'], help='Choose model type.')
    args = parser.parse_args()
    print(f"Task type is: {args.task}")

    # Load the dataset
    data_loader = DataLoader(db_path='./agri.db', table_name='farm_data')
    df = data_loader.load_data()

    # Determine the target column based on task
    if args.task == 'regression':
        target_column = 'Temperature Sensor (°C)'
    elif args.task == 'classification':
        # Special handling for classification target
        plant_type_mapping = {
            'vine crops': 'Vine Crops',
            'herbs': 'Herbs',
            'fruiting vegetables': 'Fruiting Vegetables',
            'leafy greens': 'Leafy Greens'
        }
        df['Plant Type'] = df['Plant Type'].str.strip().str.lower().map(lambda x: plant_type_mapping.get(x, x)).str.title()
        plant_stage_mapping = {
            'maturity': 'Maturity',
            'seedling': 'Seedling',
            'vegetative': 'Vegetative'
        }
        df['Plant Stage'] = df['Plant Stage'].str.strip().str.lower().map(lambda x: plant_stage_mapping.get(x, x)).str.title()
        df['Plant Type-Stage'] = df['Plant Type'] + '-' + df['Plant Stage']
        target_column = 'Plant Type-Stage'
        
    # Preprocess the data
    preprocessor = DataPreprocessor(df)
    X_train, X_test, y_train, y_test,label_encoder = preprocessor.preprocess_data(target_column=target_column, task_type=args.task)

    # Train the model
    model_trainer = ModelTrainer(model_type=args.model)
    model = model_trainer.train_model(X_train, y_train, task=args.task)

    # Evaluate the model
    evaluator = ModelEvaluator(model, task_type=args.task, label_encoder=label_encoder)
    metrics = evaluator.evaluate(X_test, y_test)

    # Print the results
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == '__main__':
    main()
