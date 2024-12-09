import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess_data(self, target_column, task_type):
        """
        Preprocess the df.
        :param target_column: The target variable.
        :param task_type: 'regression' or 'classification'.
        """
        plant_stage_mapping = {
            'maturity': 'Maturity',
            'seedling': 'Seedling',
            'vegetative': 'Vegetative'
        }

        # Normalize 'Plant Stage'
        self.df['Plant Stage'] = self.df['Plant Stage'].str.strip().str.lower().map(lambda x: plant_stage_mapping.get(x, x)).str.title()

        # Convert sensors to numeric and handle missing values
        self.df['Temperature Sensor (°C)'] = pd.to_numeric(self.df['Temperature Sensor (°C)'], errors='coerce')
        self.df['Humidity Sensor (%)'] = pd.to_numeric(self.df['Humidity Sensor (%)'], errors='coerce')
        self.df['Humidity Sensor (%)'].fillna(self.df['Humidity Sensor (%)'].mean(), inplace=True)
        self.df['Light Intensity Sensor (lux)'] = pd.to_numeric(self.df['Light Intensity Sensor (lux)'], errors='coerce')
        self.df['CO2 Sensor (ppm)'] = pd.to_numeric(self.df['CO2 Sensor (ppm)'], errors='coerce')
        self.df['EC Sensor (dS/m)'] = pd.to_numeric(self.df['EC Sensor (dS/m)'], errors='coerce')
        self.df['EC Sensor (dS/m)'].fillna(self.df['EC Sensor (dS/m)'].median(), inplace=True)
        # Filter invalid rows
        valid_rows = (self.df['Temperature Sensor (°C)'] > 0) & \
                     (self.df['Light Intensity Sensor (lux)'] >= 0) & \
                     (self.df['CO2 Sensor (ppm)'] > 0)
        self.df = self.df[valid_rows]

        # Separate features (X) and target (y)
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        # Drop columns that are not relevant to the model
        X = X.drop(columns=["System Location Code", "Previous Cycle Plant Type"])

        # Handle missing values in the target variable
        if y.isnull().sum() > 0:
            if task_type == 'regression':
                y = y.fillna(y.mean())
            elif task_type == 'classification':
                y = y.fillna(y.mode()[0])
        # Encode target variable for classification tasks
        if task_type == "classification":
            encoder = LabelEncoder()
            y = encoder.fit_transform(y)
            self.label_encoder = encoder  # Save encoder for inverse transformation if needed
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        # Handle categorical features in X
        categorical_cols = X.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            # Split the data into training and testing first
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Apply OneHotEncoder to categorical columns, fitting only on the training data
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded_train = encoder.fit_transform(X_train[categorical_cols])
            encoded_test = encoder.transform(X_test[categorical_cols])

            # Convert the encoded columns into DataFrame
            encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))
            encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

            # Drop original categorical columns and concatenate encoded columns
            X_train = X_train.drop(columns=categorical_cols).reset_index(drop=True)
            X_test = X_test.drop(columns=categorical_cols).reset_index(drop=True)

            X_train = pd.concat([X_train, encoded_train_df], axis=1)
            X_test = pd.concat([X_test, encoded_test_df], axis=1)
            if task_type == "regression":
                return X_train, X_test, y_train, y_test, None
            return X_train, X_test, y_train, y_test, self.label_encoder

