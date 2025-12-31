<h1 align="center">
  <img src="/ml_audit/ml_audit_favicon.png" alt="ML Audit Logo" width="30" style="vertical-align: middle"> 
  ML Audit
</h1>

> **Solves Data Lineage Blindness by tracking granular preprocessing steps.**

`ml-audit` is a lightweight Python library designed to bring transparency and reproducibility to data preprocessing. Unlike standard experiment trackers that treat preprocessing as a black box, this library records every granular transformation applied to your pandas DataFrame.

## Why ML Audit?

**It solves "Data Lineage Blindness".**

Most data science teams suffer from a gap in their experiment tracking:

*   **MLflow/W&B** track *metrics* (accuracy, loss) and *hyperparameters*. They often treat the cleaned dataset as a static artifact.
*   **DVC** tracks *file versions*. It tells you **that** the data changed from Version A to Version B.
*   **ML Audit** tells you **why** and **how** it changed. It logs: *"Imputed column 'Age' with mean (42.5), then Scaled with StandardScaler, then OneHotEncoded 'Gender'."*

## Features

- **Full Audit Trail**: Automatically logs every step (Imputation, Scaling, Encoding, etc.) into a JSON audit file.
- **Reproducibility**: Verify if your data pipeline produces the exact same result every time using hash validation.
- **Visualization**: Auto-generates an interactive HTML timeline of your preprocessing steps.
- **Comprehensive Operations**:
    - **Imputation**: mean, median, mode, constant, ffill, bfill.
    - **Scaling**: minmax, standard, robust, maxabs.
    - **Encoding**: onehot, label, target encoding.
    - **Balancing**: smote (via imblearn), oversample, undersample.
    - **Transformation**: log, sqrt, boxcox.
    - **Date Extraction**: Extract year, month, day from timestamps.
    - **Binning**: Discretize continuous variables into quantiles or uniform bins.
- **Multi-Column Support**: Apply operations to lists of columns efficiently.
- **Generic Support**: Track *any* arbitrary pandas method (e.g., dropna, rename).

## Installation

You can install `ml-audit` via pip:

```bash
pip install ml-audit
```

For SMOTE balancing support, install with the `balance` extra:

```bash
pip install ml-audit[balance]
```

## Interactive Demo

Try out the library instantly in your browser:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n3pGC2-r_KhGJtost8OSHhWBuCQGTv4Z?usp=sharing)


## Quick Start

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
       .balance_classes("churn", strategy='oversample') # Handles imbalanced data
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

## Detailed API Documentation

All methods support **Method Chaining** (returning `self`).

### 1. Imputation (`impute`)

Fill missing values in one or more columns using statistical strategies or specific methods.

**Signature:**
```python
auditor.impute(column, strategy='mean', fill_value=None, method=None)
```

**Parameters:**


- `column` (str | list): The column(s) to impute.
- `strategy` (str):
    - `'mean'`: Fill with mean.
    - `'median'`: Fill with median.
    - `'mode'`: Fill with mode (most frequent).
    - `'constant'`: Fill with `fill_value`.
- `method` (str):
    - `'ffill'`: Forward fill (propagates last valid observation).
    - `'bfill'`: Backward fill (use next valid observation).
    - *Note*: If `method` is provided, `strategy` is ignored.
- `fill_value` (any): Value to use when `strategy='constant'`.

**Examples:**
```python
# Impute multiple columns with median
auditor.impute(["age", "salary"], strategy='median')

# Fill with a constant value (e.g., 0)
auditor.impute("bonus", strategy='constant', fill_value=0)

# Forward fill for time-series data
auditor.impute("stock_price", method='ffill')
```

### 2. Scaling (`scale`)

Scale numerical features to a specific range or distribution.

**Signature:**
```python
auditor.scale(column, method='standard')
```

**Parameters:**

- `column` (str | list): The column(s) to scale.
- `method` (str):
    - `'standard'`: Standardize features (mean=0, std=1). Uses `StandardScaler`.
    - `'minmax'`: Scale to range [0, 1]. Uses `MinMaxScaler`.
    - `'robust'`: Scale using statistics that are robust to outliers. Uses `RobustScaler`.
    - `'maxabs'`: Scale by maximum absolute value. Uses `MaxAbsScaler`.

**Examples:**
```python
# Standardize normally distributed features
auditor.scale(["height", "weight"], method='standard')

# Normalize image pixel values
auditor.scale("pixels", method='minmax')
```

### 3. Encoding (`encode`)

Encode categorical features into numeric form.

**Signature:**
```python
auditor.encode(column, method='onehot', target_col=None)
```

**Parameters:**

- `column` (str | list): The column(s) to encode.
- `method` (str):
    - `'onehot'`: Creates binary columns for each category (e.g., `color_red`, `color_blue`).
    - `'label'`: Assigns a unique integer to each category.
    - `'target'`: Encodes categories based on the mean of the `target_col`.
- `target_col` (str): Required only for `method='target'`.

**Examples:**
```python
# One-hot encode low-cardinality nominal data
auditor.encode("color", method='onehot') 
# Result: color_red, color_blue, ...

# Label encode ordinal data
auditor.encode("quality", method='label')
# Result: 0, 1, 2...

# Target encode high-cardinality data
auditor.encode("zip_code", method='target', target_col="house_price")
```

### 4. Transformation (`transform`)

Apply mathematical transformations to columns.

**Signature:**
```python
auditor.transform(column, func='log')
```

**Parameters:**

- `column` (str | list): The column(s) to transform.
- `func` (str):
    - `'log'`: Applies `np.log1p` (natural log(1+x)). Safer for zero values.
    - `'sqrt'`: Applies square root.
    - `'cbrt'`: Applies cube root.
    - `'square'`: Squares the values.

**Examples:**
```python
# Log transform skewed data
auditor.transform("income", func='log')
```

### 5. Binning (`bin_numeric`)

Discretize continuous variables into bins (buckets).

**Signature:**
```python
auditor.bin_numeric(column, bins=5, strategy='quantile', labels=None)
```

**Parameters:**

- `column` (str | list): The column(s) to bin.
- `bins` (int): Number of bins to create.
- `strategy` (str):
    - `'quantile'`: Bins have equal number of data points (Equal Frequency).
    - `'uniform'`: Bins have equal width (Equal Width).
- `labels` (list, optional): Custom labels for the bins.

**Examples:**
```python
# Create 4 quartiles for age
auditor.bin_numeric("age", bins=4, strategy='quantile')
```

### 6. Date Extraction (`extract_date_features`)

Extract features from datetime columns.

**Signature:**
```python
auditor.extract_date_features(column, features=['year', 'month', 'day', 'weekday'])
```

**Parameters:**

- `column` (str | list): The datetime column(s).
- `features` (list): List of features to extract. Options: `'year'`, `'month'`, `'day'`, `'weekday'`, `'hour'`.

**Examples:**
```python
# Extract year and month from 'joined_date'
auditor.extract_date_features("joined_date", features=['year', 'month'])
# Creates columns: joined_date_year, joined_date_month
```

### 7. Balancing (`balance_classes`)

Balance the dataset based on the target variable.

**Signature:**
```python
auditor.balance_classes(target, strategy='oversample', random_state=42)
```

**Parameters:**

- `target` (str): The target column name (variables to predict).
- `strategy` (str):
    - `'oversample'`: Randomly duplicate samples from minority class.
    - `'undersample'`: Randomly remove samples from majority class.
    - `'smote'`: Synthetic Minority Over-sampling Technique. (Requires `imblearn`).

**Examples:**
```python
# Handle imbalanced dataset using SMOTE
auditor.balance_classes("is_fraud", strategy='smote')
```

### 8. Filtering & Dropping

Basic dataframe manipulations.

**Signatures:**
```python
auditor.filter_rows(column, operator, value)
auditor.drop_columns(columns)
```

**Examples:**
```python
# Keep only adults
auditor.filter_rows("age", ">=", 18)

# Remove PII
auditor.drop_columns(["ssn", "email"])
```

### 9. Generic Operations (`track_pandas`)

Track **any** pandas dataframe method that isn't natively built-in.

**Signature:**
```python
auditor.track_pandas(method_name, *args, **kwargs)
```

**Examples:**
```python
# Track a rename operation
auditor.track_pandas("rename", columns={"old_name": "new_name"})

# Track dropping NaNs
auditor.track_pandas("dropna", subset=["critical_col"])
```

### 10. Reproducibility

**Verify Lineage:**
You can check if re-running the recorded operations on the original data produces the exact same result (by hash).

```python
if auditor.verify_reproducibility():
    print("Pipeline is scientifically reproducible!")
else:
    print("Pipeline result mismatch!")
# Also available: auditor.replay() returns the re-computed dataframe independently
```

## Visualization

When you run `export_audit_trail()`, an HTML file is generated in the `visualizations/` folder.
This interactive timeline shows:

1.  **Step Sequence**: The order of operations.
2.  **Parameters**: What strategy/method was used (e.g., `strategy='median'`).
3.  **Data Shape**: How the row/column count changed.
4.  **Schema**: How columns were added or removed.

## License

MIT License. Free to use for personal and commercial projects.

