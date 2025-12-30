# %% [markdown]
# # Ambiguous preprocessing

# %%
import pandas as pd
import numpy as np
import json
import hashlib
from datetime import datetime
import random


# %%
np.random.seed(42)
random.seed(42)

df = pd.DataFrame({
    "age": np.random.uniform(low=18, high=100, size=1000),
    "size": np.random.uniform(low=1000, high=10000, size=1000),
    "price": np.random.randint(low=10000, high=100000, size=1000),
    "category": np.random.choice(['A', 'B', 'C'], size=1000)
})
df["bought_house"] = (
    (df["price"].fillna(df["price"].median()) > 50000) &
    (df["age"] > 40)
).astype(int)

# Add missingness to the price column
missing_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[missing_indices, 'price'] = np.nan

df.info()

# %%
df

# %%
# Stable DataFrame hashing
def hash_data(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

hash_data(df)



# %%
# Operation base class

class Operation:
    name = "base"
    
    def __init__(self):
        pass

    def apply(self, df:pd.DataFrame) -> pd.DataFrame: # “Every operation must define how it modifies a DataFrame.”
        raise NotImplementedError
    
    def serialize(self) -> dict: # “Every operation must also know how to describe itself as data.”
        raise NotImplementedError


# %%
# Concrete Operations

#  1.Load Marker
class LoadData(Operation):
    name = "load_data"

    def __init__(self, shape, columns, dtypes):
        self.shape = shape
        self.columns = columns
        self.dtypes = dtypes
    
    def apply(self, df):
        return df
    
    def serialize(self):
        return{
            "op": self.name,  # "load_data"
            "shape": self.shape,
            "columns": self.columns,
            "dtypes": self.dtypes
        }

op = LoadData(df.shape, df.columns, df.dtypes)
df_new = op.apply(df)
df_new

# %%
op.serialize()

# %%
# 2. Declarative Filter

class FilterRows(Operation):
    name = "filter_rows"

    def __init__(self, column, operator, value):
        self.column = column
        self.operator = operator
        self.value = value
    
    def apply(self, df):
        if self.operator == ">=":
            return df[df[self.column] >= self.value].reset_index(drop=True)
        if self.operator == "<=":
            return df[df[self.column] <= self.value].reset_index(drop=True)
        if self.operator == "==":
            return df[df[self.column] == self.value].reset_index(drop=True)
        raise ValueError("Unsupported operator")
    
    def serialize(self):
        return{
            'op': self.name,
            "column": self.column,
            "operator": self.operator,
            "value": self.value
        }

op2 = FilterRows('age', '>=', 50) # Filter age to aminimum of 50 
new_df = op2.apply(df)
new_df['age'].describe()


# %%
op2.serialize()

# %%
# 3.Drop Columns
class DropColumns(Operation):
    name = 'drop_columns'

    def __init__(self, columns):
        self.columns = columns
    
    def apply(self, df):
        return df.drop(columns = self.columns, axis = 1)
    
    def serialize(self):
        return{
            "op": self.name,
            "columns": self.columns           
        }

op3 = DropColumns(['size'])
df_new = op3.apply(df)
df_new.columns

# %%
op3.serialize()

# %%
# 4. Mean imputation
class ImputeMean(Operation):
    name = "impute_mean"

    def __init__(self, column, mean_value):
        self.column = column
        self.mean_value = mean_value
    
    def apply(self, df):
        df = df.copy()
        df[self.column] = df[self.column].fillna(self.mean_value)
        return df
    
    def serialize(self):
        return{
            "op": self.name,
            "column": self.column,
            "mean_value": self.mean_value
        }

op4 = ImputeMean(['price'], df['price'].mean())
df_new = op4.apply(df)
print("Before Imputation: ")
print(df.isnull().sum())
print("\nAfter Imputation: ")
print(df_new.isnull().sum())

# %%
# 5. Normalizer

class Normalization(Operation):
    name = 'normalize'

    def __init__(self, column, normalizer):
        self.column = column
        self.normalizer = normalizer
    
    def apply(self, df):
        sc = self.normalizer
        df = df.copy()
        df[self.column] = sc.fit_transform(df[self.column])
        return df
    
    def serialize(self):
        return{
            "op": self.name,
            "column": self.column,
            "normalizer": type(self.normalizer).__name__
        }

from sklearn.preprocessing import StandardScaler
op6 = Normalization(['price'], StandardScaler())
df_new = op6.apply(df)
df_new['price']

# %%
op6.serialize()

# %%
# 6. One hot encoding

class Ohe(Operation):
    name = 'one_hot_encode'

    def __init__(self, column, categories):
        self.column = column
        self.categories = categories
    
    def apply(self, df):
        df = df.copy()
        df = pd.get_dummies(df, columns=[self.column])
        return df
    
    def serialize(self):
        return{
            "op":self.name,
            "column": self.column,
            "categories": self.categories
        }
    
op7 = Ohe('category', df['category'].unique())
df_new = op7.apply(df)
df_new


# %%
op7.serialize()

# %%
# 7. Order-sensitive filter (mean-dependent)
class FilterAboveMeanPrice(Operation):
    name = "filter_above_mean_price"

    def apply(self, df):
        mean_price = df["price"].mean()
        return df[df["price"] > mean_price].reset_index(drop=True)

    def serialize(self):
        return {
            "op": self.name
        }


# %%
# Record Audit Trials

class AuditTrialRecorder:
    def __init__(self, dataframe, name = "experiment"):
        self.name = name
        self.original_df = dataframe.copy()
        self.current_df = dataframe.copy()
        self.operations = []
        self.hashes = []

        self._record(              # _record means this method is internal to the class.It is not part of the public interface.”
            LoadData(
                shape=dataframe.shape,
                columns=list(dataframe.columns),
                dtypes={c: str(t) for c, t in dataframe.dtypes.items()}
            ))
        
    def _record(self, operation: Operation):
        self.operations.append(operation) 
        self.hashes.append(hash_data(self.current_df))

    def filter_rows(self, column, operator, value):
        op = FilterRows(column, operator, value)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    def drop_columns(self, columns):
        op = DropColumns(columns)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    def impute_mean(self, column):
        mean_val = self.current_df[column].mean()
        op = ImputeMean(column, mean_val)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    def normalize_column(self, column):
        scaler = StandardScaler()
        scaler.fit(self.current_df[[column]])
        op = Normalization([column], scaler)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    def one_hot_encode(self, column):
        categories = sorted(
            pd.get_dummies(self.current_df[column], prefix=column).columns
        )
        op = Ohe(column, categories)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self
    
    def filter_above_mean_price(self):
        op = FilterAboveMeanPrice()
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    
    # Deterministic replay and verification 
    """
    Replay:
        - starts from raw data
        - reapplies every operation in order
        - produces a new dataset
    This answers: Can someone else regenerate the dataset exactly?
    """
    def replay(self):
        df = self.original_df.copy()
        for op in self.operations[1:]:
            df = op.apply(df)
        return df

    def verify_reproducibility(self):
        replayed_df = self.replay()
        return hash_data(replayed_df) == hash_data(self.current_df)
    
    # Export Audit Trial
    def export_audit_trail(self, filename=None):
        if filename is None:
            filename = "./audit_trails/audit_trial.json"
        trail = {
            "experiment": self.name,
            "created": datetime.now().isoformat(),
            "operations": [op.serialize() for op in self.operations],
            "final_hash": hash_data(self.current_df),
            "final_shape": self.current_df.shape,
            "final_columns": list(self.current_df.columns)
        }
        with open(filename, "w") as f:
            json.dump(trail, f, indent=2)
        return trail



# %%
audit = AuditTrialRecorder(df, name="house_price_data_analysis")

audit.filter_rows("age", ">=", 21)
audit.impute_mean("price")
audit.impute_mean("size")
audit.normalize_column("price")
audit.one_hot_encode("category")

print("Reproducible:", audit.verify_reproducibility())
audit.export_audit_trail()



# %% [markdown]
# ![brave_screenshot.png](attachment:brave_screenshot.png)

# %% [markdown]
# # COMPLETE EXPERIMENT: PREPROCESSING-INDUCED VARIANCE
# 
# We will produce three preprocessing paths:
# 
# - Path A: Filter → Impute → Normalize → OHE
# - Path B: Impute → Filter → Normalize → OHE
# - Path C: Filter → Normalize → Impute → OHE

# %%

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
class Modeling:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = []
    
    def train(self):
        xtrain, xtest, ytrain, ytest = train_test_split(self.X, self.y, test_size = 0.3, random_state=42)
        
        sc = StandardScaler()
        xtrain = sc.fit_transform(xtrain)
        xtest = sc.transform(xtest)

        models = [
            LogisticRegression(random_state=42),
            XGBClassifier(random_state=42),
            RandomForestClassifier(random_state=42)
            ]
        
        for i in range(len(models)):
            name = models[i].__class__.__name__
            ypred = models[i].fit(xtrain, ytrain).predict(xtest)
            evaluation = classification_report(ytest, ypred)
            accuracy = accuracy_score(ytest, ypred)
            print(f"----{models[i]}-----")
            print("Results", evaluation)
            self.results.append([name, accuracy])

data = df.copy()
# bought_house is already numeric (0 and 1), no mapping needed
# data['bought_house'] = data['bought_house'].map({
#     'yes': 1,
#     'No': 0
# })

# No need to drop rows since bought_house is already clean
# data = data.dropna(subset=['bought_house'])

data = pd.get_dummies(data, columns=['category'])
numeric_cols = data.select_dtypes(include="number").columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Convert boolean columns to integers
bool_cols = data.select_dtypes(include="bool").columns
data[bool_cols] = data[bool_cols].astype(int)

trainer = Modeling(data.drop('bought_house', axis =1), data.bought_house)
trainer.train()

# %%
results=trainer.results
print(results)

# %%
import matplotlib.pyplot as plt


#Extract names, r2s, and maes
names = [row[0] for row in results]
acuracy = [row[1] for row in results]

#R2 plot
plt.figure(figsize = (8,5))
plt.bar(names,acuracy, color = 'g')
plt.title("Accuracy")
plt.ylabel("R2")
plt.show()

# %%
# Helper: train model and return accuracy

def train_and_evaluate(df: pd.DataFrame, seed=0):
    X = df.drop(columns=["bought_house"])
    # Convert bought_house to binary: 1 for "yes"/1, 0 otherwise
    y = (df["bought_house"].astype(str).str.lower() == "yes").astype(int)
    
    # Ensure we have both classes in the dataset
    if len(y.unique()) < 2:
        return 0.5 
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    model = RandomForestClassifier(max_iter=1000, random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# %% [markdown]
# ## Define ambiguous preprocessing paths

# %%
def run_path(path_name, df):
    audit = AuditTrialRecorder(df, name=path_name)

    if path_name == "Path_A":
        audit.impute_mean("price")
        audit.filter_above_mean_price()
        audit.normalize_column("price")

    elif path_name == "Path_B":
        audit.filter_above_mean_price()
        audit.impute_mean("price")
        audit.normalize_column("price")

    elif path_name == "Path_C":
        audit.impute_mean("price")
        audit.normalize_column("price")
        audit.filter_above_mean_price()

    audit.one_hot_encode("category")

    final_df = audit.current_df

    X = final_df.drop(columns=["bought_house"])
    y = final_df["bought_house"]

    return X, y


# %%
# Use a variance-sensitive model

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def evaluate_paths(df, seeds=10):
    results = { "Path_A": [], "Path_B": [], "Path_C": [] }

    for seed in range(seeds):
        for path in results.keys():
            X, y = run_path(path, df)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed
            )

            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=seed
            )

            model.fit(X_train, y_train)
            acc = model.score(X_test, y_test)

            results[path].append(acc)

    return results

# %%
results = evaluate_paths(df)

mean_acc = {k: np.mean(v) for k, v in results.items()}
all_accs = np.concatenate(list(results.values()))

print("Mean accuracies per path:", mean_acc)
print("Overall mean accuracy:", np.mean(all_accs))
print("Variance due to preprocessing ambiguity:", np.var(all_accs))


# %%
# Visualize the preprocessing path variance

import matplotlib.pyplot as plt

# Extract accuracies for each path
path_names = list(results.keys())
path_accuracies = [results[path] for path in path_names]

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Box plot of accuracies across paths
axes[0].boxplot(path_accuracies, labels=path_names)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy Distribution by Preprocessing Path")
axes[0].grid(True, alpha=0.3)

# Plot 2: Mean accuracy per path with error bars
means = [np.mean(accs) for accs in path_accuracies]
stds = [np.std(accs) for accs in path_accuracies]
axes[1].bar(path_names, means, yerr=stds, capsize=10, alpha=0.7, color=['blue', 'green', 'red'])
axes[1].set_ylabel("Mean Accuracy")
axes[1].set_title("Mean Accuracy by Path (with Std Dev)")
axes[1].set_ylim([0.9, 1.01])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("PREPROCESSING PATH VARIANCE ANALYSIS")
print("="*60)
for path in path_names:
    accs = results[path]
    print(f"\n{path}:")
    print(f"  Mean Accuracy: {np.mean(accs):.4f}")
    print(f"  Std Dev:       {np.std(accs):.4f}")
    print(f"  Min:           {np.min(accs):.4f}")
    print(f"  Max:           {np.max(accs):.4f}")

print(f"\nOverall Variance (across all paths): {np.var(all_accs):.6f}")

# %%
import os
os.makedirs("./audit_trails", exist_ok=True)

# Create audit recorders for each path
audit_A = AuditTrialRecorder(df, name="Path_A")
audit_A.impute_mean("price")
audit_A.filter_above_mean_price()
audit_A.normalize_column("price")
audit_A.one_hot_encode("category")

audit_B = AuditTrialRecorder(df, name="Path_B")
audit_B.filter_above_mean_price()
audit_B.impute_mean("price")
audit_B.normalize_column("price")
audit_B.one_hot_encode("category")

audit_C = AuditTrialRecorder(df, name="Path_C")
audit_C.impute_mean("price")
audit_C.normalize_column("price")
audit_C.filter_above_mean_price()
audit_C.one_hot_encode("category")

# Verify reproducibility
assert audit_A.verify_reproducibility()
assert audit_B.verify_reproducibility()
assert audit_C.verify_reproducibility()

# Export audit trails
audit_A.export_audit_trail("./audit_trails/audit_A.json")
audit_B.export_audit_trail("./audit_trails/audit_B.json")
audit_C.export_audit_trail("./audit_trails/audit_C.json")

print("All audit trails verified and exported successfully!")



