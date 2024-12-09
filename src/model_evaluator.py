from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self, model, task_type, label_encoder=None):
        self.model = model
        self.task_type = task_type
        self.label_encoder = label_encoder

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model based on the task type.
        :param X_test: Features for testing.
        :param y_test: True labels for testing.
        :return: Dictionary of metrics.
        """
        y_pred = self.model.predict(X_test)
        metrics = {}

        if self.task_type == "regression":
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["r2"] = r2_score(y_test, y_pred)
        elif self.task_type == "classification":
            # Decode predictions back to original labels if needed
            if self.label_encoder:
                y_pred = self.label_encoder.inverse_transform(y_pred)
                y_test = self.label_encoder.inverse_transform(y_test)

            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["recall"] = recall_score(y_test, y_pred, average="weighted")
            metrics["precision"] = precision_score(y_test, y_pred, average="weighted")
            metrics["f1"] = f1_score(y_test, y_pred, average="weighted")

        return metrics

        return metrics
