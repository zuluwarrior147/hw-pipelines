import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import os
import typer

def load_and_preprocess(path: str = typer.Option("data"), file_name="https://storage.googleapis.com/telco-churn-123/telco-churn.csv"):
    df = pd.read_csv(file_name)
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

    nominal_cols = ['InternetService', 'Contract', 'PaymentMethod']
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    for col in binary_cols:
        df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    preprocessor = ColumnTransformer(transformers=[
        ("bin", OrdinalEncoder(), binary_cols),
        ("nom", OrdinalEncoder(), nominal_cols),
        ("num", StandardScaler(), numeric_cols)
    ], remainder='passthrough')

    X_processed = preprocessor.fit_transform(X)

    # Check if we have enough samples for stratified split
    # Need at least 2 samples per class for stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = min(class_counts)
    
    if len(y) < 10 or min_class_count < 2:
        # For small datasets, don't use stratification
        return train_test_split(X_processed, y, test_size=0.2, random_state=42)
    else:
        # For larger datasets, use stratification        
        X_train, X_test, y_train, y_test =  train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)
        os.makedirs(path, exist_ok=True)
        np.savetxt(f"{path}/X_train.csv", X_train, delimiter=",", fmt="%.8f")
        np.savetxt(f"{path}/X_test.csv", X_test, delimiter=",", fmt="%.8f")
        np.savetxt(f"{path}/y_train.csv", y_train, delimiter=",", fmt="%.8f")
        np.savetxt(f"{path}/y_test.csv", y_test, delimiter=",", fmt="%.8f")
        print(f"Data saved to {path}/X_train.csv, {path}/X_test.csv, {path}/y_train.csv, {path}/y_test.csv")

        return X_train, X_test, y_train, y_test
