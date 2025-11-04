"""
Airflow DAG for ML Pipeline with Drift Detection and Automatic Retraining

Pipeline Flow:
1. Load baseline data → Preprocess → Train initial model → Save baseline stats
2. Load production data → Preprocess → Detect drift
3. Branch: If drift detected → Retrain model → Compare models
           If no drift → Skip retraining
"""

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta
from airflow import configuration as conf

# Import our custom functions
import sys
sys.path.insert(0, '/opt/airflow')

from src.data_loader import (
    load_baseline_data,
    load_production_data,
    preprocess_data,
    preprocess_new_data
)
from src.model_trainer import (
    train_baseline_model,
    retrain_model
)
from src.drift_detector import (
    save_baseline_statistics,
    detect_drift,
    branch_on_drift
)
from src.model_evaluator import (
    compare_models,
    no_retrain_needed
)

# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Default arguments
default_args = {
    'owner': 'niranjan',
    'start_date': datetime(2024, 11, 3),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_drift_detection_pipeline',
    default_args=default_args,
    description='ML Pipeline with Drift Detection and Automatic Retraining',
    schedule_interval=None,  # Manual trigger for demo
    catchup=False,
    tags=['machine-learning', 'drift-detection', 'mlops']
)

# ==================== BASELINE MODEL TRAINING ====================

# Task 1: Load baseline data
load_baseline_task = PythonOperator(
    task_id='load_baseline_data_task',
    python_callable=load_baseline_data,
    provide_context=True,
    dag=dag,
)

# Task 2: Preprocess baseline data
preprocess_baseline_task = PythonOperator(
    task_id='preprocess_baseline_data_task',
    python_callable=preprocess_data,
    op_args=[load_baseline_task.output],
    provide_context=True,
    dag=dag,
)

# Task 3: Train baseline model
train_baseline_task = PythonOperator(
    task_id='train_baseline_model_task',
    python_callable=train_baseline_model,
    op_args=[preprocess_baseline_task.output],
    provide_context=True,
    dag=dag,
)

# Task 4: Save baseline statistics for drift detection
save_baseline_stats_task = PythonOperator(
    task_id='save_baseline_stats_task',
    python_callable=save_baseline_statistics,
    op_args=[preprocess_baseline_task.output],
    provide_context=True,
    dag=dag,
)

# ==================== PRODUCTION DATA & DRIFT DETECTION ====================

# Task 5: Load production/new data
load_production_task = PythonOperator(
    task_id='load_production_data_task',
    python_callable=load_production_data,
    provide_context=True,
    dag=dag,
)

# Task 6: Preprocess production data
preprocess_production_task = PythonOperator(
    task_id='preprocess_production_data_task',
    python_callable=preprocess_new_data,
    op_args=[load_production_task.output],
    provide_context=True,
    dag=dag,
)

# Task 7: Detect drift
detect_drift_task = PythonOperator(
    task_id='detect_drift_task',
    python_callable=detect_drift,
    op_args=[preprocess_production_task.output],
    provide_context=True,
    dag=dag,
)

# Task 8: Branch based on drift detection
branch_task = BranchPythonOperator(
    task_id='branch_on_drift_task',
    python_callable=branch_on_drift,
    provide_context=True,
    dag=dag,
)

# ==================== CONDITIONAL RETRAINING ====================

# Task 9a: Retrain model (if drift detected)
retrain_task = PythonOperator(
    task_id='retrain_model_task',
    python_callable=retrain_model,
    op_args=[preprocess_baseline_task.output, preprocess_production_task.output],
    provide_context=True,
    dag=dag,
)

# Task 10: Compare models (after retraining)
compare_models_task = PythonOperator(
    task_id='compare_models_task',
    python_callable=compare_models,
    provide_context=True,
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# Task 9b: No retrain needed (if no drift)
no_retrain_task = PythonOperator(
    task_id='no_retrain_needed_task',
    python_callable=no_retrain_needed,
    provide_context=True,
    dag=dag,
)

# Task 11: Final completion marker
completion_task = DummyOperator(
    task_id='pipeline_complete',
    trigger_rule='none_failed_min_one_success',
    dag=dag,
)

# ==================== DEFINE TASK DEPENDENCIES ====================

# Baseline model training flow
load_baseline_task >> preprocess_baseline_task
preprocess_baseline_task >> train_baseline_task
preprocess_baseline_task >> save_baseline_stats_task

# Production data processing - wait for BOTH baseline preprocessing AND stats
[preprocess_baseline_task, save_baseline_stats_task] >> load_production_task
load_production_task >> preprocess_production_task
preprocess_production_task >> detect_drift_task

# Branching logic
detect_drift_task >> branch_task

# Branch paths
branch_task >> retrain_task >> compare_models_task >> completion_task
branch_task >> no_retrain_task >> completion_task

# Compare models also needs baseline model trained
train_baseline_task >> compare_models_task

if __name__ == "__main__":
    dag.cli()