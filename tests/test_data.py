import pandas as pd
import numpy as np
import tempfile
import os
from scripts.preprocessing import load_and_preprocess

def test_load_and_preprocess():
    """Test the preprocessing function with the actual data file."""
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Check that we get the expected number of splits
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    
    # Check that X_train and X_test have the same number of features
    assert X_train.shape[1] == X_test.shape[1]
    
    # Check that y contains only binary values
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}
    
    # Check that the data is properly scaled (numeric columns should have mean close to 0)
    # Note: This is a basic check, actual values might vary
    numeric_cols_indices = list(range(16, 19))  # Indices for numeric columns
    for idx in numeric_cols_indices:
        assert abs(np.mean(X_train[:, idx])) < 1.0
        assert abs(np.mean(X_test[:, idx])) < 1.0

def test_preprocessing_with_sample_data(sample_data):
    """Test preprocessing with sample data using a temporary file."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        sample_data.to_csv(temp_file.name, index=False)
        temp_filename = temp_file.name
    
    try:
        # Test preprocessing with the temporary file
        X_train, X_test, y_train, y_test = load_and_preprocess(temp_filename)
        
        # Basic assertions
        assert X_train.shape[1] == X_test.shape[1]
        assert len(X_train) + len(X_test) == len(sample_data)
        
        # Check that y contains only binary values
        assert set(np.unique(y_train)) <= {0, 1}
        assert set(np.unique(y_test)) <= {0, 1}
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

def test_data_shape_consistency(sample_data):
    """Test that the preprocessing maintains data consistency."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        sample_data.to_csv(temp_file.name, index=False)
        temp_filename = temp_file.name
    
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess(temp_filename)
        
        # Check shapes are consistent
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        
        # Check that we have the expected number of features after preprocessing
        # 12 binary + 3 nominal + 3 numeric = 18 features
        assert X_train.shape[1] == 18
        assert X_test.shape[1] == 18
        
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

def test_preprocessing_handles_missing_values():
    """Test that preprocessing handles missing values in TotalCharges."""
    # Create data with missing TotalCharges
    data_with_missing = pd.DataFrame({
        'customerID': ['1', '2', '3', '4'],
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'Partner': ['Yes', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No', 'Yes'],
        'PhoneService': ['Yes', 'Yes', 'No', 'Yes'],
        'MultipleLines': ['No', 'Yes', 'No phone service', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'DSL', 'No'],
        'OnlineSecurity': ['No', 'Yes', 'No internet service', 'No internet service'],
        'OnlineBackup': ['Yes', 'No', 'No internet service', 'No internet service'],
        'DeviceProtection': ['No', 'Yes', 'No internet service', 'No internet service'],
        'TechSupport': ['Yes', 'No', 'No internet service', 'No internet service'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'No internet service'],
        'StreamingMovies': ['Yes', 'No', 'No internet service', 'No internet service'],
        'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        'tenure': [1, 24, 48, 12],
        'MonthlyCharges': [29.85, 56.95, 42.30, 35.00],
        'TotalCharges': [29.85, 1367.80, ' ', 420.00],  # Missing value as space
        'Churn': ['No', 'Yes', 'No', 'Yes']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        data_with_missing.to_csv(temp_file.name, index=False)
        temp_filename = temp_file.name
    
    try:
        # Should not raise an error despite missing values
        X_train, X_test, y_train, y_test = load_and_preprocess(temp_filename)
        
        # Check that we still get valid data
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_test).any()
        
    finally:
        if os.path.exists(temp_filename):
            os.unlink(temp_filename) 