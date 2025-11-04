"""
Model training and saving functions
"""

import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_baseline_model(serialized_data, **context):
    """
    Train initial baseline Random Forest model
    
    Args:
        serialized_data: Pickled preprocessed data
    
    Returns: Serialized model and metrics
    """
    try:
        logger.info("Training baseline model...")
        
        # Deserialize data
        data_dict = pickle.loads(serialized_data)
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1
        )
        
        logger.info("Fitting model...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'baseline'
        }
        
        logger.info(f"Baseline Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Save model
        model_dir = '/opt/airflow/working_data/models'
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = f"{model_dir}/baseline_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Model saved to {model_path}")
        
        # Save metrics
        metrics_path = f"{model_dir}/baseline_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Return for XCom
        output = {
            'model_path': model_path,
            'metrics': metrics,
            'feature_importance': dict(zip(
                data_dict['feature_names'],
                model.feature_importances_
            ))
        }
        
        return pickle.dumps(output)
        
    except Exception as e:
        logger.error(f"Error training baseline model: {e}")
        raise

def retrain_model(serialized_baseline, serialized_new_data, **context):
    """
    Retrain model when drift is detected
    Combines baseline and new data for retraining
    
    Args:
        serialized_baseline: Pickled baseline preprocessed data
        serialized_new_data: Pickled new preprocessed data
    
    Returns: Serialized retrained model and metrics
    """
    try:
        logger.info("Retraining model with new data...")
        
        # Deserialize data
        baseline_dict = pickle.loads(serialized_baseline)
        new_dict = pickle.loads(serialized_new_data)
        
        # Combine baseline training data with new data
        import pandas as pd
        
        X_train_combined = pd.concat([
            baseline_dict['X_train'],
            new_dict['X']
        ], ignore_index=True)
        
        y_train_combined = pd.concat([
            baseline_dict['y_train'],
            new_dict['y']
        ], ignore_index=True)
        
        # Use baseline test set for evaluation
        X_test = baseline_dict['X_test']
        y_test = baseline_dict['y_test']
        
        logger.info(f"Combined training data: {X_train_combined.shape}")
        
        # Train new model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        logger.info("Fitting retrained model...")
        model.fit(X_train_combined, y_train_combined)
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'retrained',
            'training_samples': len(X_train_combined)
        }
        
        logger.info(f"Retrained Model Performance:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Save retrained model
        model_dir = '/opt/airflow/working_data/models'
        model_path = f"{model_dir}/retrained_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Retrained model saved to {model_path}")
        
        # Save metrics
        metrics_path = f"{model_dir}/retrained_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Return for XCom
        output = {
            'model_path': model_path,
            'metrics': metrics,
            'feature_importance': dict(zip(
                baseline_dict['feature_names'],
                model.feature_importances_
            ))
        }
        
        return pickle.dumps(output)
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise