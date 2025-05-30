# Telco Customer Churn Prediction

This project implements a machine learning model to predict customer churn for a telecommunications company. The model uses logistic regression to classify whether a customer is likely to churn based on various features like service usage, contract details, and customer demographics.

## Project Structure

```
.

├── data/              # Dataset directory
├── scripts/           # Utility scripts
├── tests/             # Unit tests
├── pyproject.toml     # Modern Python project configuration
├── requirements.txt   # Main project dependencies
└── requirements-test.txt  # Test dependencies
```

## Features

- Customer demographic information
- Service usage details
- Contract information
- Payment method
- Monthly and total charges
- Tenure with the company

## Model Details

The model uses:
- Logistic Regression with L2 regularization
- Feature preprocessing including:
  - Ordinal encoding for binary features
  - One-hot encoding for categorical features
  - Standard scaling for numerical features
- Train-test split with stratification (80-20 split)

## Metrics

The model is evaluated using:
- Accuracy
- F1 Score
- ROC AUC Score

## Model Training and Storage

This project includes functionality for training models and saving them locally with metadata.

### Features

- **Model Training**: Support for Logistic Regression, Decision Tree, and XGBoost models
- **Model Storage**: Trained models are automatically saved as pickle files with metadata
- **Model Versioning**: Each trained model is saved with performance metrics and configuration
- **Local Storage**: Models are stored in the `models/` directory for easy access

### Model Storage

When training models, they are automatically saved with:

- **Metadata**: Model type, performance metrics, and configuration
- **Pickle Format**: Standard Python serialization for easy loading
- **Performance Metrics**: Accuracy, F1 score, and ROC-AUC included
- **Configuration**: Full model parameters and training settings

### Usage Examples

#### Programmatic Usage

```python
from scripts.training import train

# Train model with default configuration (logistic regression)
model, X_test, y_test, metrics = train()

# Train model with custom configuration
config = {
    'model_type': 'xgboost',
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5
}
model, X_test, y_test, metrics = train(config=config)

# Load a saved model
import pickle
with open('models/logistic_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    metrics = model_data['metrics']
    config = model_data['config']
```

### Model File Structure

Each saved model file contains:

```python
{
    'model': trained_model_object,
    'model_type': 'logistic',  # or 'tree', 'xgboost'
    'metrics': {
        'accuracy': 0.85,
        'f1_score': 0.80,
        'roc_auc': 0.90
    },
    'config': {
        'C': 1.0,
        'penalty': 'l2',
        # ... other model parameters
    }
}
```

## Testing

This project includes a comprehensive unit test suite to ensure code quality and reliability.

### Test Structure

```
tests/
├── __init__.py                 # Makes tests a proper Python package
├── conftest.py                 # Shared fixtures for all tests
├── test_preprocessing.py       # Tests for data preprocessing functionality
├── test_training.py           # Tests for model training functionality
└── test_trained_models.py     # Tests for trained model properties and performance
```

### Test Coverage

**Preprocessing Tests:**
- Data loading and preprocessing pipeline
- Handling of missing values in TotalCharges
- Data shape consistency and feature counts
- Small dataset handling (automatic stratification disabling)

**Training Tests:**
- Logistic Regression model training
- Decision Tree model training  
- XGBoost model training
- Error handling for unsupported model types
- Model saving functionality
- **Model storage** (file operations, metadata, serialization)

**Trained Model Tests:**
- Model properties and attributes validation
- Prediction capabilities (predict and predict_proba)
- Performance metrics evaluation (accuracy, F1, ROC-AUC)
- Model consistency across runs
- Feature importance availability
- Model serialization compatibility
- Edge case handling with minimal datasets
- Parameter validation and customization

### Running Tests

#### Install Test Dependencies
```bash
pip install -r requirements-test.txt
```

#### Run Tests
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=scripts --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
pytest tests/test_trained_models.py
```

### Test Features

- **97% code coverage** across the project
- **Realistic sample data** that matches the actual dataset structure
- **Temporary file handling** for safe testing without affecting real data
- **Smart stratification** handling for datasets of different sizes
- **Comprehensive edge case testing** including missing values and small datasets
- **Model validation** ensuring trained models meet quality standards
- **Model storage testing** verifying file operations and metadata functionality

### Current Coverage
- **100% coverage** for preprocessing module
- **95% coverage** for training module (including model storage functionality)
- **Fast execution** (< 3 seconds)
- **CI/CD ready** with no external dependencies during testing

## CI/CD Pipeline

This project uses CircleCI for continuous integration and deployment.

### Pipeline Features

- **Automated testing** on every pull request
- **Coverage reporting** with artifacts stored for each build
- **Python 3.9** environment with all dependencies
- **JUnit XML** test results for detailed reporting
- **Codecov integration** for coverage tracking
- **Fast execution** (typically < 3 minutes)
- **Modern Python packaging** with `pyproject.toml`

### Pipeline Configuration

The pipeline (`.circleci/config.yml`) includes:

1. **Environment Setup**: Python 3.9 Docker container
2. **Dependency Installation**: Test dependencies only (no build required)
3. **Test Execution**: Full test suite with coverage
4. **Artifact Storage**: Coverage reports and test results
5. **Coverage Upload**: Automatic upload to Codecov

### Status Checks

All pull requests must pass:
- ✅ All unit tests (27 tests)
- ✅ 97% code coverage maintained
- ✅ No linting errors
- ✅ Successful dependency installation

### Setup Instructions

To enable CircleCI for your fork:

1. Connect your GitHub repository to CircleCI
2. Add `CODECOV_TOKEN` environment variable (optional)
3. Enable status checks in GitHub branch protection rules
4. All PRs will automatically trigger the test pipeline 