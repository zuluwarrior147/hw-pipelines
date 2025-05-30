import numpy as np
import os
import tempfile
import json
from unittest.mock import patch
from scripts.training import train
from scripts.model import save_model, load_model
from xgboost import XGBClassifier

def test_xgboost_training(processed_data):
    """Test XGBoost training with default parameters."""
    X_train, X_test, y_train, y_test = processed_data
    
    with patch('numpy.loadtxt') as mock_loadtxt, \
         patch('scripts.training.save_model') as mock_save:
        
        # Configure the mock to return our test data in the right order
        mock_loadtxt.side_effect = [X_train, X_test, y_train, y_test]
        
        model, X_test_ret, y_test_ret, metrics = train()
        
        # Verify model was trained
        assert model is not None
        assert isinstance(model, XGBClassifier)
        assert hasattr(model, 'predict')
        
        # Verify data is returned correctly
        assert X_test_ret is not None
        assert y_test_ret is not None
        np.testing.assert_array_equal(X_test_ret, X_test)
        np.testing.assert_array_equal(y_test_ret, y_test)
        
        # Verify metrics are calculated
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        
        # Verify save_model was called
        mock_save.assert_called_once_with(model, metrics)

def test_train_with_provided_data(processed_data):
    """Test training with provided data instead of loading from files."""
    X_train, X_test, y_train, y_test = processed_data
    
    with patch('scripts.training.save_model') as mock_save:
        model, X_test_ret, y_test_ret, metrics = train(data=(X_train, X_test, y_train, y_test))
        
        # Verify model was trained
        assert model is not None
        assert isinstance(model, XGBClassifier)
        assert hasattr(model, 'predict')
        
        # Verify the same test data is returned
        np.testing.assert_array_equal(X_test_ret, X_test)
        np.testing.assert_array_equal(y_test_ret, y_test)
        
        # Verify metrics are calculated
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Verify save_model was called
        mock_save.assert_called_once_with(model, metrics)

def test_model_parameters(processed_data):
    """Test that the model is created with the expected parameters."""
    X_train, X_test, y_train, y_test = processed_data
    
    with patch('scripts.training.save_model'):
        model, _, _, _ = train(data=(X_train, X_test, y_train, y_test))
        
        # Verify model parameters match the expected values
        assert model.n_estimators == 150
        assert model.learning_rate == 0.2
        assert model.max_depth == 5
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.567
        assert model.scale_pos_weight == 3
        # Note: use_label_encoder and eval_metric are not accessible as attributes after training

def test_model_predictions(processed_data):
    """Test that the trained model can make predictions."""
    X_train, X_test, y_train, y_test = processed_data
    
    with patch('scripts.training.save_model'):
        model, X_test_ret, y_test_ret, metrics = train(data=(X_train, X_test, y_train, y_test))
        
        # Test predict method
        predictions = model.predict(X_test_ret)
        assert len(predictions) == len(y_test_ret)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Test predict_proba method
        probabilities = model.predict_proba(X_test_ret)
        assert probabilities.shape == (len(y_test_ret), 2)
        assert all(0 <= prob <= 1 for row in probabilities for prob in row)
        assert all(abs(sum(row) - 1.0) < 1e-6 for row in probabilities)

def test_save_model_function(processed_data):
    """Test the save_model function with JSON format."""
    X_train, X_test, y_train, y_test = processed_data
    
    # Create and train a model
    model = XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    metrics = {"accuracy": 0.85, "f1_score": 0.80, "roc_auc": 0.90}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_filepath = os.path.join(temp_dir, "test_model")
        
        # Test saving model
        save_model(model, metrics, base_filepath)
        
        # Verify files were created
        model_file = f"{base_filepath}.json"
        metadata_file = f"{base_filepath}_metadata.json"
        assert os.path.exists(model_file)
        assert os.path.exists(metadata_file)
        
        # Load and verify the saved metadata
        with open(metadata_file, 'r') as f:
            saved_metadata = json.load(f)
        
        assert 'model_type' in saved_metadata
        assert 'metrics' in saved_metadata
        assert 'model_file' in saved_metadata
        
        assert saved_metadata['model_type'] == "xgboost"
        assert saved_metadata['metrics'] == metrics
        assert saved_metadata['model_file'] == model_file
        
        # Verify the saved model can be loaded and make predictions
        loaded_model = XGBClassifier()
        loaded_model.load_model(model_file)
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == len(y_test)

def test_save_model_default_path(processed_data):
    """Test save_model with default filepath."""
    X_train, X_test, y_train, y_test = processed_data
    
    # Create and train a model
    model = XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    metrics = {"accuracy": 0.85, "f1_score": 0.80, "roc_auc": 0.90}
    
    with patch('os.makedirs') as mock_makedirs, \
         patch('builtins.open', create=True) as mock_open, \
         patch.object(model, 'save_model') as mock_model_save, \
         patch('json.dump') as mock_json_dump:
        
        save_model(model, metrics)
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with("models", exist_ok=True)
        
        # Verify model save was called
        mock_model_save.assert_called_once_with("models/model.json")
        
        # Verify metadata file opening
        mock_open.assert_called_once_with("models/model_metadata.json", 'w')
        
        # Verify JSON dump was called
        mock_json_dump.assert_called_once()
        
        # Verify the data structure passed to json.dump
        call_args = mock_json_dump.call_args[0]
        saved_metadata = call_args[0]
        assert saved_metadata['model_type'] == "xgboost"
        assert saved_metadata['metrics'] == metrics
        assert saved_metadata['model_file'] == "models/model.json"

def test_load_model_function(processed_data):
    """Test the load_model function."""
    X_train, X_test, y_train, y_test = processed_data
    
    # Create and train a model
    model = XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    metrics = {"accuracy": 0.85, "f1_score": 0.80, "roc_auc": 0.90}
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_filepath = os.path.join(temp_dir, "test_model")
        
        # Save model first
        save_model(model, metrics, base_filepath)
        
        # Load model and verify
        loaded_model, loaded_metadata = load_model(base_filepath)
        
        # Verify loaded metadata
        assert loaded_metadata['model_type'] == "xgboost"
        assert loaded_metadata['metrics'] == metrics
        
        # Verify loaded model works
        original_predictions = model.predict(X_test)
        loaded_predictions = loaded_model.predict(X_test)
        
        # Predictions should be identical
        np.testing.assert_array_equal(original_predictions, loaded_predictions)

def test_load_model_default_path():
    """Test load_model with default filepath."""
    metadata = {
        'model_type': 'xgboost',
        'metrics': {'accuracy': 0.85},
        'model_file': 'models/model.json'
    }
    
    mock_model = XGBClassifier()
    
    with patch('builtins.open', create=True) as mock_open, \
         patch('json.load', return_value=metadata) as mock_json_load, \
         patch.object(XGBClassifier, 'load_model') as mock_model_load:
        
        mock_model_load.return_value = None  # load_model modifies the object in place
        
        loaded_model, loaded_metadata = load_model()
        
        # Verify metadata file was opened
        mock_open.assert_called_once_with("models/model_metadata.json", 'r')
        
        # Verify JSON load was called
        mock_json_load.assert_called_once()
        
        # Verify model load was called
        mock_model_load.assert_called_once_with('models/model.json')
        
        # Verify returned data
        assert loaded_metadata == metadata

def test_metrics_calculation(processed_data):
    """Test that metrics are calculated correctly."""
    X_train, X_test, y_train, y_test = processed_data
    
    with patch('scripts.training.save_model'):
        model, X_test_ret, y_test_ret, metrics = train(data=(X_train, X_test, y_train, y_test))
        
        # Manually calculate metrics to verify
        predictions = model.predict(X_test_ret)
        probabilities = model.predict_proba(X_test_ret)[:, 1]
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        expected_accuracy = accuracy_score(y_test_ret, predictions)
        expected_f1 = f1_score(y_test_ret, predictions)
        expected_roc_auc = roc_auc_score(y_test_ret, probabilities)
        
        # Verify metrics match expected values
        assert abs(metrics['accuracy'] - expected_accuracy) < 1e-10
        assert abs(metrics['f1_score'] - expected_f1) < 1e-10
        assert abs(metrics['roc_auc'] - expected_roc_auc) < 1e-10

def test_training_integration(processed_data):
    """Test the complete training pipeline without mocking."""
    X_train, X_test, y_train, y_test = processed_data
    
    # Test the actual training function end-to-end
    with patch('numpy.loadtxt') as mock_loadtxt:
        mock_loadtxt.side_effect = [X_train, X_test, y_train, y_test]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Temporarily change the models directory
            original_save_model = save_model
            
            def mock_save_model(model, metrics, filepath=None):
                if filepath is None:
                    filepath = os.path.join(temp_dir, "model")
                return original_save_model(model, metrics, filepath)
            
            with patch('scripts.training.save_model', side_effect=mock_save_model):
                model, X_test_ret, y_test_ret, metrics = train()
                
                # Verify everything worked
                assert isinstance(model, XGBClassifier)
                assert len(X_test_ret) > 0
                assert len(y_test_ret) > 0
                assert all(key in metrics for key in ['accuracy', 'f1_score', 'roc_auc'])
                
                # Verify model files were created
                model_file = os.path.join(temp_dir, "model.json")
                metadata_file = os.path.join(temp_dir, "model_metadata.json")
                assert os.path.exists(model_file)
                assert os.path.exists(metadata_file) 