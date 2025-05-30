import os
import json
from xgboost import XGBClassifier
import wandb
import typer


def save_model(model, metrics, filepath="models"):
    """
    Save model to file with metadata in JSON format.
    
    Args:
        model: Trained XGBoost model object
        metrics (dict): Performance metrics
        filepath (str): Optional custom filepath for saving the model (without extension)
    """
    os.makedirs(filepath, exist_ok=True)
    
    # Save model using XGBoost's native format
    model_filepath = f"{filepath}/model.json"
    model.save_model(model_filepath)
    
    # Save metadata as JSON
    metadata = {
        'model_type': 'xgboost',
        'metrics': metrics,
        'model_file': model_filepath
    }
    
    metadata_filepath = f"{filepath}/metadata.json"
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metrics: {metrics}")
    print(f"Model saved to {model_filepath}")
    print(f"Metadata saved to {metadata_filepath}")


def load_model(filepath: str = typer.Option("models")):
    """
    Load model and metadata from JSON files.
    
    Args:
        filepath (str): Optional custom filepath (without extension)
    
    Returns:
        tuple: (model, metadata)
    """
    
    # Load metadata
    metadata_filepath = f"{filepath}/metadata.json"
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = XGBClassifier()
    model.load_model(metadata['model_file'])
    print(f"Model {metadata['model_type']} loaded from {metadata['model_file']} with metrics {metadata['metrics']}")
    
    return model, metadata


def upload_model(model_path: str = typer.Option("models")):
    _, metadata = load_model(model_path)

    run = wandb.init()

    artifact = wandb.Artifact(
        name="telco-churn-model",
        type="model",
        description=f"Trained {metadata['model_type']} model for churn prediction",
        metadata={
            "model_type": metadata["model_type"],
            "accuracy": metadata["metrics"]["accuracy"],
            "f1_score": metadata["metrics"]["f1_score"],
            "roc_auc": metadata["metrics"]["roc_auc"]
        }
    )

    artifact.add_file(metadata['model_file'])
    
    wandb.log_artifact(artifact)


if __name__ == "__main__":
    upload_model()
