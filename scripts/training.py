import numpy as np  
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
from pathlib import Path
import typer

# Add project root to Python path for clean absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.model import save_model

def train(data=None, data_path: str = typer.Option("data"), model_path: str = typer.Option("models")):
    """
    Train a model and return it along with test data for evaluation.
    
    Args:
        data (tuple): Optional tuple of (X_train, X_test, y_train, y_test)
    
    Returns:
        tuple: (trained_model, X_test, y_test, metrics)
    """
    if data is None:
        X_train = np.loadtxt(f"{data_path}/X_train.csv", delimiter=",")
        X_test = np.loadtxt(f"{data_path}/X_test.csv", delimiter=",")
        y_train = np.loadtxt(f"{data_path}/y_train.csv", delimiter=",")
        y_test = np.loadtxt(f"{data_path}/y_test.csv", delimiter=",")
    else:
        X_train, X_test, y_train, y_test = data
    
    model_params = {
            'n_estimators': 150,
            'learning_rate': 0.2,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.567,
            'scale_pos_weight': 3,
            'use_label_encoder': False,
            'eval_metric': "logloss"
    }
    model = XGBClassifier(**model_params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probas)
    }

    # Save model with metrics
    save_model(model, metrics, model_path)
    
    return model, X_test, y_test, metrics

if __name__ == "__main__":
    train()