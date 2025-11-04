"""
Model evaluation and comparison functions
"""

import pickle
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_models(**context):
    """
    Compare baseline and retrained model performance
    Determine which model to use in production
    
    Returns: Serialized comparison results
    """
    try:
        logger.info("Comparing model performances...")
        
        # Load baseline metrics
        baseline_metrics_path = '/opt/airflow/working_data/models/baseline_metrics.json'
        with open(baseline_metrics_path, 'r') as f:
            baseline_metrics = json.load(f)
        
        # Load retrained metrics
        retrained_metrics_path = '/opt/airflow/working_data/models/retrained_metrics.json'
        with open(retrained_metrics_path, 'r') as f:
            retrained_metrics = json.load(f)
        
        # Load drift report
        drift_report_path = '/opt/airflow/working_data/reports/drift_report.json'
        with open(drift_report_path, 'r') as f:
            drift_report = json.load(f)
        
        # Calculate improvements
        comparison = {
            'baseline_metrics': baseline_metrics,
            'retrained_metrics': retrained_metrics,
            'improvements': {
                'accuracy': retrained_metrics['accuracy'] - baseline_metrics['accuracy'],
                'precision': retrained_metrics['precision'] - baseline_metrics['precision'],
                'recall': retrained_metrics['recall'] - baseline_metrics['recall'],
                'f1_score': retrained_metrics['f1_score'] - baseline_metrics['f1_score']
            },
            'drift_info': {
                'drift_detected': drift_report['overall_drift_detected'],
                'drift_score': drift_report['drift_score'],
                'features_with_drift': len(drift_report['significant_drifts'])
            },
            'recommendation': ''
        }
        
        # Determine recommendation
        if comparison['improvements']['f1_score'] > 0.01:  # 1% improvement threshold
            comparison['recommendation'] = 'Use retrained model - significant improvement'
            comparison['selected_model'] = 'retrained'
        elif comparison['improvements']['f1_score'] > -0.01:
            comparison['recommendation'] = 'Models perform similarly - use retrained for recency'
            comparison['selected_model'] = 'retrained'
        else:
            comparison['recommendation'] = 'Retrained model worse - investigate data quality issues'
            comparison['selected_model'] = 'baseline'
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MODEL COMPARISON RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Baseline Model:")
        logger.info(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {baseline_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {baseline_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {baseline_metrics['f1_score']:.4f}")
        logger.info(f"\nRetrained Model:")
        logger.info(f"  Accuracy:  {retrained_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {retrained_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {retrained_metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {retrained_metrics['f1_score']:.4f}")
        logger.info(f"\nImprovements:")
        logger.info(f"  Accuracy:  {comparison['improvements']['accuracy']:+.4f}")
        logger.info(f"  Precision: {comparison['improvements']['precision']:+.4f}")
        logger.info(f"  Recall:    {comparison['improvements']['recall']:+.4f}")
        logger.info(f"  F1-Score:  {comparison['improvements']['f1_score']:+.4f}")
        logger.info(f"\n📊 Recommendation: {comparison['recommendation']}")
        logger.info(f"{'='*60}\n")
        
        # Save comparison report
        comparison_path = '/opt/airflow/working_data/reports/model_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Comparison report saved to {comparison_path}")
        
        return pickle.dumps(comparison)
        
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise

def no_retrain_needed(**context):
    """
    Placeholder task when no retraining is needed
    """
    logger.info("✓ No retraining needed - baseline model is still valid")
    logger.info("Continuing to use baseline model for predictions")
    
    return "baseline_model_active"