# Airflow Lab 1 - ML Pipeline with Drift Detection & Automatic Retraining

## Project Overview

This project implements Apache Airflow pipeline that trains a machine learning model, monitors production data for distribution drift, and automatically retrains the model when significant drift is detected.

### Key Features

-  **Baseline Model Training** - Random Forest classifier on bank marketing data
-  **Drift Detection** - Statistical tests (KS, Chi-square, PSI) to detect data drift
-  **Automatic Retraining** - Conditional workflow triggers retraining when drift exceeds threshold
-  **Model Comparison** - Performance evaluation before and after retraining
-  **Production-Ready** - Dockerized Airflow pipeline with proper error handling

---

##  ML Model

This pipeline is designed for **binary classification** to predict term deposit subscriptions in bank marketing campaigns. It incorporates drift detection to monitor data distribution changes over time and automatically retrains the model when needed.

### Dataset: Bank Marketing (UCI ML Repository)

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Task:** Predict if a client will subscribe to a term deposit (yes/no)
- **Features:** 17 features including age, job, balance, campaign data
- **Size:** ~41,000 samples
- **Target Distribution:** Imbalanced (~88% no, ~12% yes)

### Prerequisites

Before running this pipeline, ensure you have the following installed:

- **Docker Desktop** (with at least 4GB RAM allocated)
- **Python 3.8+** (for data download script)
- **Required Python packages:** pandas, scikit-learn, scipy, numpy, ucimlrepo

---

## Quick Start

### Step 1: Download Dataset

```bash
# Install required package
pip install ucimlrepo pandas

# Run data download script
python data_downloader.py
```

This will:
- Download the Bank Marketing dataset from UCI
- Create temporal splits: baseline (60%), production (20%), retrain (20%)
- Save files to `data/` directory

### Step 2: Set Up Airflow Environment

```bash
# Create environment file
cat > .env << EOF
AIRFLOW_UID=50000
AIRFLOW_PROJ_DIR=.
_AIRFLOW_WWW_USER_USERNAME=airflow2
_AIRFLOW_WWW_USER_PASSWORD=airflow2
EOF

# Initialize Airflow database
docker compose up airflow-init

# Start Airflow services
docker compose up
```

Wait for health check: `GET /health HTTP/1.1" 200`

### Step 3: Access Airflow UI

1. Open browser: `http://localhost:8080`
2. Login credentials: **airflow2** / **airflow2**
3. Find DAG: `ml_drift_detection_pipeline`
4. Enable DAG (toggle switch)
5. Trigger DAG (play button ▶️)

---

## 📂 Project Structure

```
airflow_lab/
├── dags/
│   └── drift_detection_pipeline_dag.py    # Main Airflow DAG
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Data loading & preprocessing
│   ├── model_trainer.py                   # Model training & retraining
│   ├── drift_detector.py                  # Drift detection logic
│   └── model_evaluator.py                 # Model comparison
├── data/
│   ├── baseline_data.csv                  # Training data (60%)
│   ├── production_data.csv                # Production data (20%)
│   └── retrain_data.csv                   # Additional data (20%)
├── working_data/                          # Generated during execution
│   ├── models/
│   │   ├── baseline_model.pkl
│   │   ├── retrained_model.pkl
│   │   └── *.json (metrics)
│   ├── baseline_stats/
│   │   └── baseline_statistics.pkl
│   └── reports/
│       ├── drift_report.json
│       └── model_comparison.json
├── logs/                                  # Airflow logs
├── plugins/                               # Airflow plugins
├── docker-compose.yaml                    # Docker configuration
├── .env                                   # Environment variables
├── data_downloader.py                     # Dataset download script
└── README.md
```

---

##  Pipeline Architecture

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    BASELINE MODEL TRAINING                      │
├─────────────────────────────────────────────────────────────────┤
│  1. load_baseline_data_task                                     │
│  2. preprocess_baseline_data_task                               │
│  3. train_baseline_model_task                                   │
│  4. save_baseline_stats_task                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCTION DATA MONITORING                    │
├─────────────────────────────────────────────────────────────────┤
│  5. load_production_data_task                                   │
│  6. preprocess_production_data_task                             │
│  7. detect_drift_task                                           │
│  8. branch_on_drift_task                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
              YES (Drift)         NO (No Drift)
                    │                   │
                    ↓                   ↓
┌──────────────────────────┐   ┌───────────────────────┐
│  9. retrain_model_task   │   │  no_retrain_needed   │
│ 10. compare_models_task  │   └───────────────────────┘
└──────────────────────────┘            │
                    │                   │
                    └─────────┬─────────┘
                              ↓
                    ┌──────────────────┐
                    │ pipeline_complete│
                    └──────────────────┘
```

### Task Descriptions

#### Baseline Training Tasks

1. **load_baseline_data_task**
   - Loads baseline CSV data
   - Serializes data for XCom transfer
   - Logs dataset shape and target distribution

2. **preprocess_baseline_data_task**
   - Encodes categorical variables using LabelEncoder
   - Splits data into train/test sets (80/20)
   - Identifies numerical vs categorical features

3. **train_baseline_model_task**
   - Trains Random Forest Classifier
   - Model parameters: 100 trees, max_depth=10, balanced class weights
   - Evaluates on test set and saves metrics

4. **save_baseline_stats_task**
   - Calculates baseline statistics for all features
   - Stores distributions for drift detection
   - Saves to `working_data/baseline_stats/`

#### Production Monitoring Tasks

5. **load_production_data_task**
   - Loads production/new data
   - Simulates incoming production data

6. **preprocess_production_data_task**
   - Applies same encoding as baseline (consistency)
   - Handles unseen categories
   - Prepares data for drift detection

7. **detect_drift_task**
   - **Numerical features:** Kolmogorov-Smirnov test + PSI
   - **Categorical features:** Chi-square test
   - Calculates overall drift score
   - Saves detailed drift report

8. **branch_on_drift_task**
   - Decision point: Drift detected?
   - Threshold: 30% of features OR avg PSI > 0.2
   - Routes to retrain or skip

#### Conditional Retraining Tasks

9. **retrain_model_task** (if drift detected)
   - Combines baseline + production data
   - Trains new model with updated data
   - Saves retrained model and metrics

10. **compare_models_task**
    - Compares baseline vs retrained performance
    - Calculates improvement metrics
    - Provides recommendation

---

##  Drift Detection Methodology

### Statistical Tests Used

#### 1. Population Stability Index (PSI)
- **Purpose:** Measures distribution shift in numerical features
- **Threshold:** 
  - PSI < 0.1: No significant change
  - 0.1 ≤ PSI < 0.2: Moderate change
  - PSI ≥ 0.2: Significant drift ⚠️

#### 2. Kolmogorov-Smirnov (KS) Test
- **Purpose:** Compares cumulative distributions
- **Threshold:** p-value < 0.05 indicates drift

#### 3. Chi-Square Test
- **Purpose:** Tests independence of categorical distributions
- **Threshold:** p-value < 0.05 indicates drift

### Drift Decision Logic

```python
overall_drift_detected = (
    drift_percentage > 30% OR 
    avg_PSI > 0.2
)
```

If drift detected → Trigger automatic retraining

---

##  Key Functions

### 1. Data Loading (`src/data_loader.py`)

```python
def load_baseline_data(**context):
    """Load baseline data for initial model training"""
    # Returns: Serialized data dictionary
```

```python
def preprocess_data(serialized_data, **context):
    """Encode categorical variables, split train/test"""
    # Returns: Serialized preprocessed data with encoders
```

### 2. Model Training (`src/model_trainer.py`)

```python
def train_baseline_model(serialized_data, **context):
    """Train initial Random Forest model"""
    # Returns: Model path, metrics, feature importance
```

```python
def retrain_model(serialized_baseline, serialized_new_data, **context):
    """Retrain model when drift detected"""
    # Returns: Retrained model path and updated metrics
```

### 3. Drift Detection (`src/drift_detector.py`)

```python
def detect_drift(serialized_new_data, **context):
    """Detect drift using statistical tests"""
    # Returns: Drift report with detailed analysis
```

```python
def branch_on_drift(**context):
    """Branch decision based on drift detection"""
    # Returns: Next task ID (retrain or skip)
```

### 4. Model Evaluation (`src/model_evaluator.py`)

```python
def compare_models(**context):
    """Compare baseline and retrained model performance"""
    # Returns: Comparison report with recommendations
```

---

## ⚙️ Docker Configuration

### docker-compose.yaml

Key configurations:
- **Executor:** LocalExecutor (parallel task execution)
- **Database:** PostgreSQL 13
- **Python Packages:** pandas, scikit-learn, scipy, numpy
- **XCom Pickling:** Enabled for data passing
- **Volume Mounts:** dags, src, data, working_data, logs

### Environment Variables (.env)

```bash
AIRFLOW_UID=50000                      # User ID for permissions
AIRFLOW_PROJ_DIR=.                     # Project directory
_AIRFLOW_WWW_USER_USERNAME=airflow2    # Admin username
_AIRFLOW_WWW_USER_PASSWORD=airflow2    # Admin password
```

---

##  Expected Results

### Baseline Model Performance

```
Accuracy:  ~0.91
Precision: ~0.35
Recall:    ~0.89
F1-Score:  ~0.50
```

### Drift Detection Results

```
Overall Drift Detected: YES 
Drift Score (Avg PSI): 0.168
Features with Drift: 16/16 (100.0%)

Significant Drifts:
- day_of_week: PSI=0.842 (Very High)
- campaign: PSI=0.155 (Moderate)
- age: PSI=0.066 (Low but KS p-value < 0.05)
- All categorical features show significant Chi-square p-values
```

### After Retraining

```
Retrained Model Performance:
  Accuracy:  ~0.91
  Precision: ~0.35
  Recall:    ~0.89
  F1-Score:  ~0.50

Improvements:
  F1-Score: +0.001 (Models perform similarly)

Recommendation: Use retrained model for recency
```

---

## 📸 Pipeline Execution Screenshots

The following screenshots demonstrate the successful execution of the drift detection pipeline:

### DAG Overview
![DAG Overview](Assets/Airflow_lab_1(1).png)

*Figure 1: Airflow DAGs page showing ml_drift_detection_pipeline - Active status with successful execution*

### Complete Pipeline Graph
![Pipeline Graph](Assets/Airflow_lab_1(2).png)

*Figure 2: Complete pipeline workflow with all 12 tasks executed successfully (all green boxes)*

### Gantt Chart View
![Simple](Assets/Airflow_lab_1(3).png)

*Figure 3: Gantt chart showing task execution timeline and duration*

---

## 🔍 Key Observations from Execution

### Drift Detection Results
- **All 16 features showed drift** - Temporal data split successfully simulated production scenario
- **Highest PSI**: day_of_week (0.842) - Indicates strong seasonal patterns
- **Automatic trigger worked** - Branch operator correctly routed to retraining path

### Model Performance
- **Baseline F1-Score**: 0.4985
- **Retrained F1-Score**: 0.4995
- **Improvement**: +0.0010 (Models perform similarly, used retrained for recency)

### Pipeline Reliability
- Zero task failures in production run
- XCom data passing functioning correctly
- Error handling prevented any crashes

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **MLOps Pipeline Design** - End-to-end workflow automation with Airflow
2. **Drift Detection** - Statistical methods for monitoring data quality in production
3. **Conditional Workflows** - Branch operations and task dependencies in Airflow
4. **Model Lifecycle Management** - Training, monitoring, and retraining workflows
5. **Docker Orchestration** - Containerized ML pipeline deployment
6. **XCom Communication** - Data passing between Airflow tasks
7. **Error Handling** - Production-ready exception handling and logging

## 📄 License

This project is for educational purposes as part of IE7374 MLOps coursework at Northeastern University.

---
