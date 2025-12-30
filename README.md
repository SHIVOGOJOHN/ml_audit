# ML Audit 📊

> **Track, Audit, and Visualize your Machine Learning Preprocessing Pipelines.**

`ml-audit` is a lightweight Python library designed to bring transparency and reproducibility to data preprocessing. It records every transformation applied to your pandas DataFrame, ensures scientific reproducibility, and automatically generates beautiful HTML visualizations of your data lineage.

## ✨ Features

- **Full Audit Trail**: Automatically logs every step (Imputation, Scaling, Encoding, etc.) into a JSON audit file.
- **Reproducibility**: Verify if your data pipeline produces the exact same result every time.
- **Visualization**: Auto-generates an interactive HTML timeline of your preprocessing steps.
- **Comprehensive Operations**:
    - **Imputation**: `mean`, `median`, `mode`, `constant`, `ffill`, `bfill`.
    - **Scaling**: `minmax`, `standard`, `robust`, `maxabs`.
    - **Encoding**: `onehot`, `label`, `target` encoding.
    - **Balancing**: `smote` (via imblearn), `oversample` (random), `undersample`.
    - **Transformation**: `log`, `sqrt`, `boxcox`, etc.
    - **Date Extraction**: Extract year, month, day, etc. from timestamps.
- **Multi-Column Support**: Apply operations to lists of columns efficiently.
- **Generic Support**: Track *any* arbitrary pandas method (e.g., `dropna`, `rename`).

## 🚀 Installation

You can install `ml-audit` via pip:

```bash
pip install ml-audit
```

For SMOTE balancing support, install with the `balance` extra:

```bash
pip install ml-audit[balance]
```

## 📖 Quick Start

### 1. Initialize the Recorder

```python
import pandas as pd
from ml_audit import AuditTrialRecorder

# Load your data
df = pd.read_csv("data.csv")

# Initialize the auditor wrapped around your dataframe
auditor = AuditTrialRecorder(df, name="experiment_v1")
```

### 2. Apply Preprocessing

Chain methods fluently. Operations are applied immediately to `auditor.current_df`.

```python
auditor.filter_rows("age", ">=", 18) \
       .impute(["salary", "score"], strategy='median') \
       .scale(["salary", "age"], method='minmax') \
       .encode("gender", method='onehot') \
       .balance_classes("churn", strategy='oversample') # Hands imbalanced data
```

### 3. Access Data

```python
processed_df = auditor.current_df
print(processed_df.head())
```

### 4. Export & Visualize

Save the audit trail. This will generate a JSON file (`audit_trails/`) and an HTML visualization (`visualizations/`).

```python
auditor.export_audit_trail("audit.json")
# Output:
# - audit_trails/audit.json
# - visualizations/audit.html
```

## 📚 Detailed Documentation

### Multi-Column Operations
All major preprocessing methods accept either a single string or a list of strings for column names.

```python
# Scale multiple columns at once
auditor.scale(["height", "weight", "bmi"], method='standard')
```

### Generic Pandas Tracking
For operations not natively built-in, use `track_pandas` to record any DataFrame method.

```python
# Track a rename operation
auditor.track_pandas("rename", columns={"old_name": "new_name"})

# Track dropping NaNs
auditor.track_pandas("dropna", subset=["critical_col"])
```

### Reproducibility Check
Verify that replaying your logs produces the exact same data hash as the current state.

```python
if auditor.verify_reproducibility():
    print("Pipeline is scientifically reproducible! ✅")
else:
    print("Pipeline result mismatch! ❌")
```

## 🎨 Visualization
Open the generated HTML file in `visualizations/` to see a timeline like this:

- **Step 1: Load Data** (Shape: 1000x5)
- **Step 2: Impute** (`salary` -> median)
- **Step 3: Scale** (`age` -> minmax)
- ...

## 📄 License
MIT License. Free to use for personal and commercial projects.
