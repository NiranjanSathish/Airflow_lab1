"""
Script to download and prepare the Bank Marketing dataset using ucimlrepo
Run this once to set up your data directory

Installation: pip install ucimlrepo pandas scikit-learn
"""

import pandas as pd
import os

def download_and_prepare_data():
    """Download and prepare the Bank Marketing dataset"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Downloading Bank Marketing dataset from UCI ML Repository...")
    
    try:
        # Import here to give clear error message if not installed
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset 
        print("Fetching dataset (ID: 222)...")
        bank_marketing = fetch_ucirepo(id=222)
        
        # Get data as pandas dataframes
        X = bank_marketing.data.features
        y = bank_marketing.data.targets
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        print("\n✓ Dataset downloaded successfully!")
        print(f"\n📊 Dataset Metadata:")
        print(f"  Name: {bank_marketing.metadata.get('name', 'Bank Marketing')}")
        print(f"  Description: {bank_marketing.metadata.get('abstract', 'N/A')[:100]}...")
        
        print(f"\n📋 Variable Information:")
        if hasattr(bank_marketing, 'variables'):
            print(bank_marketing.variables.head())
        
        print(f"\n Dataset Info:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Features: {df.columns.tolist()}")
        print(f"  - Target distribution:\n{df['y'].value_counts()}")
        print(f"  - Target percentage: {df['y'].value_counts(normalize=True) * 100}")
        
        # Create temporal splits to simulate drift
        # We'll split data into 3 periods: baseline, production, retrain
        n = len(df)
        
        baseline_data = df.iloc[:int(n*0.6)]  # First 60% - baseline period
        production_data = df.iloc[int(n*0.6):int(n*0.8)]  # Next 20% - drift period
        retrain_data = df.iloc[int(n*0.8):]  # Last 20% - retrain data
        
        # Save splits
        baseline_data.to_csv('data/baseline_data.csv', index=False)
        production_data.to_csv('data/production_data.csv', index=False)
        retrain_data.to_csv('data/retrain_data.csv', index=False)
        
        print(f"\n✓ Data splits created:")
        print(f"  - Baseline: {len(baseline_data)} rows")
        print(f"  - Production: {len(production_data)} rows")
        print(f"  - Retrain: {len(retrain_data)} rows")
        
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        print("\nAlternative: Download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/Bank+Marketing")
        return False

if __name__ == "__main__":
    download_and_prepare_data()