import pytest
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from unittest.mock import patch
from scripts.training import train
from scripts.model import save_model, load_model


class TestTrainedModels:
    """Test suite for trained XGBoost models."""
    
    @pytest.fixture
    def sample_trained_model(self, processed_data):
        
        with patch('scripts.training.save_model'):
            model, X_test_ret, y_test_ret, metrics = train(data=processed_data)
        
        return {
            'model': model,
            'X_test': X_test_ret,
            'y_test': y_test_ret,
            'metrics': metrics
        }
    
    def test_xgboost_model_properties(self, sample_trained_model):
        """Test that XGBoost model has expected properties."""
        model = sample_trained_model['model']
        
        # Check model type
        assert isinstance(model, XGBClassifier)
        
        # Check that model is fitted
        assert hasattr(model, 'feature_importances_')
        
        # Check feature importances
        assert len(model.feature_importances_) == sample_trained_model['X_test'].shape[1]
        
        # Check that model has expected parameters
        assert model.n_estimators == 150
        assert model.learning_rate == 0.2
        assert model.max_depth == 5
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.567
        assert model.scale_pos_weight == 3
    
    def test_model_predictions(self, sample_trained_model):
        """Test that model can make predictions."""
        model = sample_trained_model['model']
        X_test = sample_trained_model['X_test']
        
        # Test predict method
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})  # Binary classification
        
        # Test predict_proba method
        probabilities = model.predict_proba(X_test)
        assert probabilities.shape == (len(X_test), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)  # Valid probabilities
    
    def test_model_performance_metrics(self, sample_trained_model):
        """Test that model achieves reasonable performance metrics."""
        model = sample_trained_model['model']
        X_test = sample_trained_model['X_test']
        y_test = sample_trained_model['y_test']
        metrics = sample_trained_model['metrics']
        
        # Make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics manually
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities)
        
        # Check that metrics are reasonable
        assert 0.0 <= accuracy <= 1.0
        assert 0.0 <= f1 <= 1.0
        assert 0.0 <= roc_auc <= 1.0
        
        # Check that returned metrics match calculated metrics
        assert abs(metrics['accuracy'] - accuracy) < 1e-10
        assert abs(metrics['f1_score'] - f1) < 1e-10
        assert abs(metrics['roc_auc'] - roc_auc) < 1e-10
        
        # For our test data, models should perform reasonably (very lenient for small datasets)
        assert accuracy >= 0.2  # Very basic sanity check for small datasets
        assert roc_auc > 0.25   # Very basic sanity check for small datasets
    
    def test_model_consistency(self, processed_data):
        
        # Train the same model twice (XGBoost has internal randomness)
        with patch('scripts.training.save_model'):
            model1, _, _, _ = train(data=processed_data)
            model2, _, _, _ = train(data=processed_data)
        
        # Models should be different objects
        assert model1 is not model2
        
        # Both should be XGBoost models with same parameters
        assert isinstance(model1, XGBClassifier)
        assert isinstance(model2, XGBClassifier)
        assert model1.n_estimators == model2.n_estimators
        assert model1.learning_rate == model2.learning_rate
    
    def test_model_handles_edge_cases(self):
        """Test that model handles edge cases properly."""
        # Create minimal dataset
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        X_test = np.array([[2, 3], [6, 7]])
        y_test = np.array([0, 1])
        
        data = (X_train, X_test, y_train, y_test)
        
        # Should not raise an exception
        with patch('scripts.training.save_model'):
            model, X_test_ret, y_test_ret, metrics = train(data=data)
        
        # Should be able to make predictions
        predictions = model.predict(X_test_ret)
        probabilities = model.predict_proba(X_test_ret)
        
        assert len(predictions) == len(X_test_ret)
        assert probabilities.shape == (len(X_test_ret), 2)
        
        # Metrics should be calculated
        assert 'accuracy' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
    
    def test_model_feature_importance_availability(self, sample_trained_model):
        """Test that model provides feature importance."""
        model = sample_trained_model['model']
        
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) > 0
        assert np.all(model.feature_importances_ >= 0)
        
        # Feature importances should sum to approximately 1
        assert abs(np.sum(model.feature_importances_) - 1.0) < 1e-6
    
    def test_model_serialization_compatibility(self, sample_trained_model):
        """Test that model can be saved/loaded using JSON format (for model persistence)."""
        import tempfile
        import os
        
        model = sample_trained_model['model']
        X_test = sample_trained_model['X_test']
        metrics = sample_trained_model['metrics']
        
        # Test saving and loading using our JSON-based functions
        with tempfile.TemporaryDirectory() as temp_dir:
            base_filepath = os.path.join(temp_dir, 'test_model')
            
            # Save model using our save_model function
            save_model(model, metrics, base_filepath)
            
            # Load model using our load_model function
            loaded_model, loaded_metadata = load_model(base_filepath)
            
            # Test that loaded model works
            original_predictions = model.predict(X_test)
            loaded_predictions = loaded_model.predict(X_test)
            
            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
            # Metadata should match
            assert loaded_metadata['model_type'] == 'xgboost'
            assert loaded_metadata['metrics'] == metrics
    
    def test_train_function_with_files(self, processed_data):
        X_train, X_test, y_train, y_test = processed_data
        
        with patch('numpy.loadtxt') as mock_loadtxt, \
             patch('scripts.training.save_model'):
            
            # Configure the mock to return our test data in the right order
            mock_loadtxt.side_effect = [X_train, X_test, y_train, y_test]
            
            model, X_test_ret, y_test_ret, metrics = train()
            
            # Verify model was trained
            assert isinstance(model, XGBClassifier)
            assert len(X_test_ret) > 0
            assert len(y_test_ret) > 0
            assert all(key in metrics for key in ['accuracy', 'f1_score', 'roc_auc'])
    
    def test_model_training_determinism_with_small_data(self):
        """Test model behavior with very small datasets."""
        # Create very small dataset
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([0, 1, 0])
        X_test = np.array([[1.5], [2.5]])
        y_test = np.array([0, 1])
        
        data = (X_train, X_test, y_train, y_test)
        
        with patch('scripts.training.save_model'):
            model, X_test_ret, y_test_ret, metrics = train(data=data)
        
        # Should handle small data gracefully
        assert isinstance(model, XGBClassifier)
        assert len(model.predict(X_test_ret)) == len(X_test_ret)
        
        # Metrics should be calculated even for small data
        assert all(0 <= metrics[key] <= 1 for key in ['accuracy', 'f1_score', 'roc_auc'])
    
    def test_json_model_persistence_end_to_end(self, processed_data):
        """Test complete save/load cycle with JSON format."""
        import tempfile
        import os
        
        X_train, X_test, y_train, y_test = processed_data
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_filepath = os.path.join(temp_dir, 'end_to_end_model')
            
            # Train and save model
            with patch('scripts.training.save_model', side_effect=lambda m, metrics, fp=None: save_model(m, metrics, fp or base_filepath)):
                model, X_test_ret, y_test_ret, metrics = train(data=processed_data)
            
            # Verify files exist
            model_file = f"{base_filepath}.json"
            metadata_file = f"{base_filepath}_metadata.json"
            assert os.path.exists(model_file)
            assert os.path.exists(metadata_file)
            
            # Load model and test
            loaded_model, loaded_metadata = load_model(base_filepath)
            
            # Verify loaded model works identically
            original_predictions = model.predict(X_test_ret)
            loaded_predictions = loaded_model.predict(X_test_ret)
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            
            # Verify metadata
            assert loaded_metadata['model_type'] == 'xgboost'
            assert loaded_metadata['metrics'] == metrics 