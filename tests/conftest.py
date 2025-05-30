import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = {
        'customerID': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
        'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'No', 'Yes', 'No phone service', 'No', 'Yes', 'No phone service', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No', 'DSL', 'Fiber optic', 'DSL', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service', 'No internet service', 'No', 'Yes', 'No internet service', 'No internet service', 'No', 'Yes'],
        'OnlineBackup': ['Yes', 'No', 'No internet service', 'No internet service', 'Yes', 'No', 'No internet service', 'No internet service', 'Yes', 'No'],
        'DeviceProtection': ['No', 'Yes', 'No internet service', 'No internet service', 'No', 'Yes', 'No internet service', 'No internet service', 'No', 'Yes'],
        'TechSupport': ['Yes', 'No', 'No internet service', 'No internet service', 'Yes', 'No', 'No internet service', 'No internet service', 'Yes', 'No'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'No internet service', 'No', 'Yes', 'No internet service', 'No internet service', 'No', 'Yes'],
        'StreamingMovies': ['Yes', 'No', 'No internet service', 'No internet service', 'Yes', 'No', 'No internet service', 'No internet service', 'Yes', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'],
        'tenure': [1, 24, 48, 12, 6, 36, 18, 60, 3, 30],
        'MonthlyCharges': [29.85, 56.95, 42.30, 35.00, 25.50, 65.20, 45.75, 80.10, 20.25, 55.40],
        'TotalCharges': [29.85, 1367.80, 2030.40, 420.00, 153.00, 2347.20, 823.50, 4806.00, 60.75, 1662.00],
        'Churn': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

@pytest.fixture
def processed_data():
    """Create processed data for model testing."""
    X_train = np.random.randn(100, 19)
    X_test = np.random.randn(20, 19)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)
    return X_train, X_test, y_train, y_test 