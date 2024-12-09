from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

class ModelTrainer:
    def __init__(self, model_type='linear_regression'):
        self.model_type = model_type

    def train_model(self, X_train, y_train, task):
        # Select model based on the model type
        if task == "regression":
            if self.model_type == 'linear_regression':
                model = LinearRegression()
            elif self.model_type == 'random_forest':
                model = RandomForestRegressor()
            elif self.model_type == 'xgboost':
                model = XGBRegressor()
        else:
            if self.model_type == 'linear_regression':
                model = LogisticRegression()
            if self.model_type == 'random_forest':
                model = RandomForestClassifier()
            elif self.model_type == 'xgboost':
                model = XGBClassifier()
            else: 
                print(f"Unable to perform on model: {self.model_type} and task: {task}")

        # Train the model
        model.fit(X_train, y_train)
        return model
