"""
Data loading and preprocessing functions for the drift detection pipeline
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_baseline_data(**context):
    """
    Load baseline data for initial model training
    Returns: Serialized data dictionary
    """
    try:
        logger.info("Loading baseline data...")
        
        # Load data
        df = pd.read_csv('/opt/airflow/data/baseline_data.csv')
        
        logger.info(f"Loaded baseline data: {df.shape}")
        logger.info(f"Target distribution:\n{df['y'].value_counts()}")
        
        # Serialize and return via XCom
        data_dict = {
            'data': df,
            'shape': df.shape,
            'target_dist': df['y'].value_counts().to_dict()
        }
        
        return pickle.dumps(data_dict)
        
    except Exception as e:
        logger.error(f"Error loading baseline data: {e}")
        raise

def load_production_data(**context):
    """
    Load production/new data for drift detection
    Returns: Serialized data dictionary
    """
    try:
        logger.info("Loading production data...")
        
        # Load data
        df = pd.read_csv('/opt/airflow/data/production_data.csv')
        
        logger.info(f"Loaded production data: {df.shape}")
        logger.info(f"Target distribution:\n{df['y'].value_counts()}")
        
        # Serialize and return via XCom
        data_dict = {
            'data': df,
            'shape': df.shape,
            'target_dist': df['y'].value_counts().to_dict()
        }
        
        return pickle.dumps(data_dict)
        
    except Exception as e:
        logger.error(f"Error loading production data: {e}")
        raise

def preprocess_data(serialized_data, **context):
    """
    Preprocess data: encode categorical variables, split features/target
    
    Args:
        serialized_data: Pickled data dictionary from load functions
    
    Returns: Serialized preprocessed data
    """
    try:
        logger.info("Preprocessing data...")
        
        # Deserialize input
        data_dict = pickle.loads(serialized_data)
        df = data_dict['data']
        
        # Separate features and target
        X = df.drop('y', axis=1)
        y = df['y'].map({'no': 0, 'yes': 1})
        
        # Identify categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logger.info(f"Categorical features: {categorical_cols}")
        logger.info(f"Numerical features: {numerical_cols}")
        
        # Encode categorical variables
        label_encoders = {}
        X_encoded = X.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        
        # Prepare output
        processed_dict = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoders': label_encoders,
            'categorical_cols': categorical_cols,
            'numerical_cols': numerical_cols,
            'feature_names': X_encoded.columns.tolist()
        }
        
        return pickle.dumps(processed_dict)
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def preprocess_new_data(serialized_data, **context):
    """
    Preprocess new/production data using the same encoding as baseline
    Pulls label encoders from XCom to ensure consistency
    
    Args:
        serialized_data: Pickled data dictionary from load_production_data
    
    Returns: Serialized preprocessed data
    """
    try:
        logger.info("Preprocessing production data...")
        
        # Deserialize input
        data_dict = pickle.loads(serialized_data)
        df = data_dict['data']
        
        # Get label encoders from baseline preprocessing task
        ti = context['ti']
        baseline_processed_bytes = ti.xcom_pull(task_ids='preprocess_baseline_data_task')
        
        # Check if we got data
        if baseline_processed_bytes is None:
            logger.error("Failed to get baseline preprocessing data from XCom")
            raise ValueError("Baseline preprocessing data not found in XCom")
        
        baseline_processed = pickle.loads(baseline_processed_bytes)
        label_encoders = baseline_processed['label_encoders']
        categorical_cols = baseline_processed['categorical_cols']
        
        # Separate features and target
        X = df.drop('y', axis=1)
        y = df['y'].map({'no': 0, 'yes': 1})
        
        # Encode using baseline encoders
        X_encoded = X.copy()
        
        for col in categorical_cols:
            le = label_encoders[col]
            # Handle unseen categories by mapping to -1
            X_encoded[col] = X[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        
        logger.info(f"Production data preprocessed: {X_encoded.shape}")
        
        # Prepare output
        processed_dict = {
            'X': X_encoded,
            'y': y,
            'raw_X': X,  # Keep raw for drift detection
            'feature_names': X_encoded.columns.tolist()
        }
        
        return pickle.dumps(processed_dict)
        
    except Exception as e:
        logger.error(f"Error preprocessing production data: {e}")
        raise