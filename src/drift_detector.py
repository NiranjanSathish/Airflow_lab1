"""
Drift detection using statistical tests and PSI (Population Stability Index)
"""

import pickle
import numpy as np
import pandas as pd
from scipy import stats
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def calculate_psi(baseline_arr, production_arr, bins=10):
    """
    Calculate Population Stability Index (PSI)
    
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change (drift detected)
    
    Args:
        baseline_arr: Baseline feature values
        production_arr: Production feature values
        bins: Number of bins for discretization
    
    Returns: PSI value
    """
    try:
        # Handle edge cases
        if len(baseline_arr) == 0 or len(production_arr) == 0:
            return 0.0
        
        # Create bins based on baseline distribution
        breakpoints = np.percentile(baseline_arr, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) <= 1:
            return 0.0
        
        # Calculate distributions
        baseline_counts = np.histogram(baseline_arr, bins=breakpoints)[0]
        production_counts = np.histogram(production_arr, bins=breakpoints)[0]
        
        # Convert to percentages (avoid division by zero)
        baseline_pct = baseline_counts / len(baseline_arr) + 1e-10
        production_pct = production_counts / len(production_arr) + 1e-10
        
        # Calculate PSI
        psi = np.sum((production_pct - baseline_pct) * np.log(production_pct / baseline_pct))
        
        return psi
        
    except Exception as e:
        logger.warning(f"Error calculating PSI: {e}")
        return 0.0

def save_baseline_statistics(serialized_data, **context):
    """
    Calculate and save baseline statistics for drift detection
    
    Args:
        serialized_data: Pickled baseline preprocessed data
    
    Returns: Serialized baseline statistics
    """
    try:
        logger.info("Calculating baseline statistics...")
        
        # Deserialize data
        data_dict = pickle.loads(serialized_data)
        X_train = data_dict['X_train']
        numerical_cols = data_dict['numerical_cols']
        categorical_cols = data_dict['categorical_cols']
        
        baseline_stats = {
            'numerical_stats': {},
            'categorical_stats': {},
            'feature_names': data_dict['feature_names'],
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols
        }
        
        # Calculate statistics for numerical features
        for col in numerical_cols:
            baseline_stats['numerical_stats'][col] = {
                'mean': float(X_train[col].mean()),
                'std': float(X_train[col].std()),
                'min': float(X_train[col].min()),
                'max': float(X_train[col].max()),
                'median': float(X_train[col].median()),
                'values': X_train[col].values.tolist()  # Store for PSI calculation
            }
        
        # Calculate statistics for categorical features
        for col in categorical_cols:
            value_counts = X_train[col].value_counts(normalize=True).to_dict()
            baseline_stats['categorical_stats'][col] = {
                'distribution': value_counts,
                'unique_count': int(X_train[col].nunique()),
                'values': X_train[col].values.tolist()  # Store for chi-square test
            }
        
        # Save to file
        stats_dir = '/opt/airflow/working_data/baseline_stats'
        os.makedirs(stats_dir, exist_ok=True)
        
        stats_path = f"{stats_dir}/baseline_statistics.pkl"
        with open(stats_path, 'wb') as f:
            pickle.dump(baseline_stats, f)
        
        logger.info(f"Baseline statistics saved to {stats_path}")
        
        return pickle.dumps(baseline_stats)
        
    except Exception as e:
        logger.error(f"Error saving baseline statistics: {e}")
        raise

def detect_drift(serialized_new_data, **context):
    """
    Detect drift by comparing new data with baseline statistics
    Uses KS test for numerical features and Chi-square for categorical
    
    Args:
        serialized_new_data: Pickled new preprocessed data
    
    Returns: Branch task ID based on drift detection
    """
    try:
        logger.info("Detecting drift in production data...")
        
        # Load baseline statistics
        stats_path = '/opt/airflow/working_data/baseline_stats/baseline_statistics.pkl'
        with open(stats_path, 'rb') as f:
            baseline_stats = pickle.load(f)
        
        # Deserialize new data
        new_dict = pickle.loads(serialized_new_data)
        X_new = new_dict['X']
        
        drift_results = {
            'numerical_drift': {},
            'categorical_drift': {},
            'overall_drift_detected': False,
            'drift_score': 0.0,
            'significant_drifts': []
        }
        
        psi_threshold = 0.2  # Significant drift threshold
        ks_threshold = 0.05  # p-value threshold for KS test
        
        total_psi = 0
        drift_count = 0
        
        # Check numerical features
        logger.info("Checking numerical features for drift...")
        for col in baseline_stats['numerical_cols']:
            baseline_values = np.array(baseline_stats['numerical_stats'][col]['values'])
            production_values = X_new[col].values
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, production_values)
            
            # Calculate PSI
            psi = calculate_psi(baseline_values, production_values)
            total_psi += psi
            
            drift_detected = (psi >= psi_threshold) or (ks_pvalue < ks_threshold)
            
            drift_results['numerical_drift'][col] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'psi': float(psi),
                'drift_detected': bool(drift_detected),  # Convert to Python bool
                'baseline_mean': baseline_stats['numerical_stats'][col]['mean'],
                'production_mean': float(production_values.mean())
            }
            
            if drift_detected:
                drift_count += 1
                drift_results['significant_drifts'].append({
                    'feature': col,
                    'type': 'numerical',
                    'psi': float(psi),
                    'ks_pvalue': float(ks_pvalue)
                })
                logger.warning(f"  ⚠️  Drift detected in {col}: PSI={psi:.4f}, KS p-value={ks_pvalue:.4f}")
            else:
                logger.info(f"  ✓ No drift in {col}: PSI={psi:.4f}")
        
        # Check categorical features
        logger.info("Checking categorical features for drift...")
        for col in baseline_stats['categorical_cols']:
            baseline_values = np.array(baseline_stats['categorical_stats'][col]['values'])
            production_values = X_new[col].values
            
            # Chi-square test
            try:
                # Create contingency table
                baseline_counts = pd.Series(baseline_values).value_counts()
                production_counts = pd.Series(production_values).value_counts()
                
                # Align indices
                all_categories = sorted(set(baseline_counts.index) | set(production_counts.index))
                baseline_aligned = np.array([baseline_counts.get(cat, 0) for cat in all_categories])
                production_aligned = np.array([production_counts.get(cat, 0) for cat in all_categories])
                
                # Normalize to same total (fix chi-square sum mismatch)
                baseline_normalized = baseline_aligned / baseline_aligned.sum() * production_aligned.sum()
                
                # Add small constant to avoid zero division
                baseline_normalized = baseline_normalized + 1
                production_aligned = production_aligned + 1
                
                chi2_stat, chi2_pvalue = stats.chisquare(production_aligned, baseline_normalized)
                
                drift_detected = chi2_pvalue < ks_threshold
                
                drift_results['categorical_drift'][col] = {
                    'chi2_statistic': float(chi2_stat),
                    'chi2_pvalue': float(chi2_pvalue),
                    'drift_detected': bool(drift_detected)  # Convert to Python bool
                }
                
                if drift_detected:
                    drift_count += 1
                    drift_results['significant_drifts'].append({
                        'feature': col,
                        'type': 'categorical',
                        'chi2_pvalue': float(chi2_pvalue)
                    })
                    logger.warning(f"  ⚠️  Drift detected in {col}: Chi2 p-value={chi2_pvalue:.4f}")
                else:
                    logger.info(f"  ✓ No drift in {col}")
                    
            except Exception as e:
                logger.warning(f"Could not perform chi-square test for {col}: {str(e)[:100]}")
        
        # Calculate overall drift score
        num_features = len(baseline_stats['numerical_cols']) + len(baseline_stats['categorical_cols'])
        drift_results['drift_score'] = float(total_psi / max(len(baseline_stats['numerical_cols']), 1))
        drift_results['drift_percentage'] = float((drift_count / num_features) * 100 if num_features > 0 else 0)
        
        # Overall drift decision: if more than 30% of features show drift OR avg PSI > 0.2
        drift_results['overall_drift_detected'] = bool(
            drift_results['drift_percentage'] > 30 or 
            drift_results['drift_score'] > psi_threshold
        )
        
        # Save drift report
        report_dir = '/opt/airflow/working_data/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = f"{report_dir}/drift_report.json"
        with open(report_path, 'w') as f:
            json.dump(make_json_serializable(drift_results), f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"DRIFT DETECTION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Drift Detected: {drift_results['overall_drift_detected']}")
        logger.info(f"Drift Score (Avg PSI): {drift_results['drift_score']:.4f}")
        logger.info(f"Features with Drift: {drift_count}/{num_features} ({drift_results['drift_percentage']:.1f}%)")
        logger.info(f"Drift Report saved to: {report_path}")
        logger.info(f"{'='*60}\n")
        
        # Push to XCom for branching decision
        context['ti'].xcom_push(key='drift_detected', value=drift_results['overall_drift_detected'])
        context['ti'].xcom_push(key='drift_score', value=drift_results['drift_score'])
        
        return pickle.dumps(drift_results)
        
    except Exception as e:
        logger.error(f"Error detecting drift: {e}")
        raise

def branch_on_drift(**context):
    """
    Branch operator function to decide whether to retrain
    
    Returns: Task ID to execute next
    """
    ti = context['ti']
    drift_detected = ti.xcom_pull(key='drift_detected', task_ids='detect_drift_task')
    
    if drift_detected:
        logger.info("🔄 Drift detected! Branching to retrain model...")
        return 'retrain_model_task'
    else:
        logger.info("✓ No significant drift detected. Skipping retraining.")
        return 'no_retrain_needed_task'