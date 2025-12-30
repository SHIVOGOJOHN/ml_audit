# %%
import pandas as pd
import numpy as np
import random
from ml_audit import AuditTrialRecorder

# %%
# Create dummy data
df = pd.DataFrame({
    "age": np.random.uniform(low=18, high=100, size=1000),
    "size": np.random.uniform(low=1000, high=10000, size=1000),
    "price": np.random.randint(low=10000, high=100000, size=1000),
    "bought_house": random.choices(['yes', 'No'], weights=[0.4, 0.6], k=1000),
    "category": np.random.choice(['A', 'B', 'C'], size=1000)
})

# Add missingness
missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[missing_indices, 'price'] = np.nan

# %%
# Initialize Recorder
audit = AuditTrialRecorder(df, name="lib_refactor_experiment")

# %%
# Apply Operations (Mix of specific and generic)
audit.filter_rows("age", ">=", 21)\
     .impute_mean("price")\
     .normalize_column("price")\
     .one_hot_encode("category")\
     .track_pandas("dropna") # Generic operation example!

# %%
# Verify and Export
print("Reproducible:", audit.verify_reproducibility())
trail = audit.export_audit_trail("lib_audit_trial.json")
print("Audit trail exported to lib_audit_trial.json")
