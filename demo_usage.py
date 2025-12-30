# Required imports
import pandas as pd
import numpy as np
from ml_audit import AuditTrialRecorder

def run_demo():
    print("--- Starting ML Audit Comprehensive Demo ---")

    # 1. Create a Sample Dataset
    data = {
        'employee_id': range(101, 116),
        'department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 'Marketing', 'Sales', 'HR', 'IT', 'Sales', 'IT', 'IT', 'IT', 'HR', 'Marketing'],
        'gender': ['M', 'F', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'M', 'F', 'M', 'F', 'M', 'F'],
        'salary': [50000, 85000, 45000, np.nan, 92000, 60000, 52000, 48000, np.nan, 55000, 88000, 91000, 42000, 49000, 62000],
        'performance_score': [3.5, 4.2, 3.8, 2.9, 4.8, 4.0, 3.2, 3.9, 4.5, 3.1, 4.7, 4.6, 3.0, 3.9, 4.1],
        'years_experience': [2, 8, 3, 1, 10, 5, 2, 4, 9, 3, 9, 10, 2, 4, 5],
        'churn': [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1] # Imbalanced target
    }
    df = pd.DataFrame(data)
    
    print("\n[1] Original Data Shape:", df.shape)
    print("Class Dist (Churn):", df['churn'].value_counts().to_dict())

    # 2. Initialize
    auditor = AuditTrialRecorder(df, name="comprehensive_experiment")

    # 3. Apply Comprehensive Pipeline
    auditor.filter_rows("performance_score", ">=", 2.0) \
           .impute(["salary", "performance_score"], strategy='median') \
           .scale(["salary", "performance_score"], method='minmax') \
           .encode("gender", method='label') \
           .extract_date_features("years_experience") \
           .bin_numeric("performance_score", bins=3, strategy='quantile', labels=['Low','Med','High']) \
           .balance_classes("churn", strategy='oversample') # Balancing step

    # 4. Results
    processed_df = auditor.current_df
    
    print("\n[2] Processed Data Sample:")
    print(processed_df.head())
    print("Final Shape:", processed_df.shape)
    print("Class Dist (Churn):", processed_df['churn'].value_counts().to_dict())

    # 5. Verify & Export
    auditor.export_audit_trail("comprehensive_audit.json")
    print(f"\n[3] Audit trail saved to 'comprehensive_audit.json'")

if __name__ == "__main__":
    run_demo()
