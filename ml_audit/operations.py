from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class Operation(ABC):
    name = "base"
    
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def serialize(self) -> dict:
        pass

class LoadData(Operation):
    name = "load_data"

    def __init__(self, shape, columns, dtypes):
        self.shape = shape
        self.columns = columns
        self.dtypes = dtypes
    
    def apply(self, df):
        return df
    
    def serialize(self):
        return {
            "op": self.name,
            "shape": self.shape,
            "columns": self.columns,
            "dtypes": self.dtypes
        }

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
        if self.operator == ">":
            return df[df[self.column] > self.value].reset_index(drop=True)
        if self.operator == "<":
            return df[df[self.column] < self.value].reset_index(drop=True)
        
        raise ValueError(f"Unsupported operator: {self.operator}")
    
    def serialize(self):
        return {
            'op': self.name,
            "column": self.column,
            "operator": self.operator,
            "value": self.value
        }

class DropColumns(Operation):
    name = 'drop_columns'

    def __init__(self, columns):
        self.columns = columns
    
    def apply(self, df):
        return df.drop(columns=self.columns, axis=1)
    
    def serialize(self):
        return {
            "op": self.name,
            "columns": self.columns           
        }

# --- Expanded Preprocessing Operations ---

class Impute(Operation):
    name = "impute"

    def __init__(self, column, strategy='mean', fill_value=None, method=None):
        self.column = column # Can be list or single
        self.strategy = strategy
        self.fill_value = fill_value
        self.method = method # 'ffill', 'bfill'
        self.stats_ = {} # Store computed stats for reproducibility
    
    def apply(self, df: pd.DataFrame):
        df = df.copy()
        cols = [self.column] if isinstance(self.column, str) else self.column
        
        # Method based (ffill/bfill)
        if self.method:
            for col in cols:
                df[col] = df[col].fillna(method=self.method)
            return df

        # Strategy based
        if self.strategy == 'constant':
            for col in cols:
                df[col] = df[col].fillna(self.fill_value)
            self.stats_['value'] = self.fill_value
            return df

        for col in cols:
            val = None
            if self.strategy == 'mean':
                val = df[col].mean()
            elif self.strategy == 'median':
                val = df[col].median()
            elif self.strategy == 'mode':
                val = df[col].mode()[0]
            
            if val is not None:
                df[col] = df[col].fillna(val)
                self.stats_[col] = val
                
        return df
    
    def serialize(self):
        return {
            "op": self.name,
            "column": self.column,
            "strategy": self.strategy,
            "method": self.method,
            "fill_value": self.fill_value,
            "statistics": str(self.stats_) # For reference
        }

class Scale(Operation):
    name = "scale"
    
    def __init__(self, column, method='standard', scaler=None):
        self.column = column
        self.method = method
        self.scaler = scaler # Store the fitted scaler sklearn object if possible
    
    def apply(self, df):
        df = df.copy()
        cols = [self.column] if isinstance(self.column, str) else self.column
        
        # We re-import here to avoid hard dependency at top level if desired, 
        # but typical ML usage assumes sklearn.
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
        
        if self.scaler is None:
            if self.method == 'standard': self.scaler = StandardScaler()
            elif self.method == 'minmax': self.scaler = MinMaxScaler()
            elif self.method == 'robust': self.scaler = RobustScaler()
            elif self.method == 'maxabs': self.scaler = MaxAbsScaler()
            else: raise ValueError(f"Unknown scaling method: {self.method}")
            
            self.scaler.fit(df[cols])
            
        df[cols] = self.scaler.transform(df[cols])
        return df
        
    def serialize(self):
        return {
            "op": self.name,
            "column": self.column,
            "method": self.method,
            # Serialize params roughly (perfect serialization requires pickle which is not JSON)
            "params": str(self.scaler.get_params()) if self.scaler else {}
        }

class Encode(Operation):
    name = "encode"
    
    def __init__(self, column, method='onehot', target_col=None, mapping_=None):
        self.column = column # Can be list or single string
        self.method = method
        self.target_col = target_col
        self.mapping_ = mapping_ # Store mapping for label/target encoding
        self.categories_ = None # For OneHot
        
    def apply(self, df):
        df = df.copy()
        cols = [self.column] if isinstance(self.column, str) else self.column
        
        if self.method == 'onehot':
            # get_dummies handles list of columns natively
            df = pd.get_dummies(df, columns=cols, prefix=cols)
            # Store new columns for record (approximate, since we don't know exact output names easily without checking)
            new_cols = []
            for c in df.columns:
                for col in cols:
                    if c.startswith(f"{col}_"):
                        new_cols.append(c)
            self.categories_ = list(set(new_cols))
            
        elif self.method == 'label':
            from sklearn.preprocessing import LabelEncoder
            if self.mapping_ is None: self.mapping_ = {}
            
            for col in cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.mapping_[col] = list(le.classes_)
            
        elif self.method == 'target':
            if self.target_col is None:
                raise ValueError("Target column required for target encoding")
            
            if self.mapping_ is None: self.mapping_ = {}
            
            # Simple Mean Target Encoding per column
            for col in cols:
                if col not in self.mapping_:
                     # Fix for FutureWarning: The default value of numeric_only in DataFrameGroupBy.mean is deprecated.
                     self.mapping_[col] = df.groupby(col)[self.target_col].mean().to_dict()
                
                # Map and fill unknowns with global mean
                global_mean = df[self.target_col].mean()
                df[col] = df[col].map(self.mapping_[col]).fillna(global_mean)
            
        return df

    def serialize(self):
        return {
            "op": self.name,
            "column": self.column,
            "method": self.method,
            "target": self.target_col,
            "mapping_sample": str(self.mapping_)[:100] if self.mapping_ else None 
        }

class Transform(Operation):
    name = "transform"
    
    def __init__(self, column, func='log'):
        self.column = column
        self.func = func
    
    def apply(self, df):
        df = df.copy()
        cols = [self.column] if isinstance(self.column, str) else self.column
        
        for col in cols:
            if self.func == 'log':
                df[col] = np.log1p(df[col]) # Log1p is safer for 0
            elif self.func == 'sqrt':
                df[col] = np.sqrt(df[col])
            elif self.func == 'cbrt':
                df[col] = np.cbrt(df[col])
            elif self.func == 'square':
                df[col] = np.square(df[col])
        return df
    
    def serialize(self):
        return {
            "op": self.name,
            "column": self.column,
            "func": self.func
        }

class Discretize(Operation):
    name = "discretize"
    
    def __init__(self, column, bins=5, strategy='quantile', labels=None):
        self.column = column
        self.bins = bins
        self.strategy = strategy
        self.labels = labels
        self.bin_edges_ = {}
    
    def apply(self, df):
        df = df.copy()
        cols = [self.column] if isinstance(self.column, str) else self.column
        
        for col in cols:
            # Using pandas qcut/cut
            if self.strategy == 'quantile':
                df[col], edges = pd.qcut(df[col], q=self.bins, labels=self.labels, retbins=True)
            else: # uniform
                df[col], edges = pd.cut(df[col], bins=self.bins, labels=self.labels, retbins=True)
            self.bin_edges_[col] = list(edges)
            
        return df

    def serialize(self):
        return {
            "op": self.name,
            "column": self.column,
            "bins": self.bins,
            "strategy": self.strategy,
            "edges": self.bin_edges_
        }

class DateExtract(Operation):
    name = "date_extract"
    
    def __init__(self, column, features=['year', 'month', 'day']):
        self.column = column
        self.features = features
    
    def apply(self, df):
        df = df.copy()
        cols = [self.column] if isinstance(self.column, str) else self.column
        
        for col in cols:
            try:
                dt_col = pd.to_datetime(df[col])
            except Exception:
                 # If it fails (e.g. it's already extracted numeric), skip
                continue

            for feat in self.features:
                new_col_name = f"{col}_{feat}"
                if feat == 'year': df[new_col_name] = dt_col.dt.year
                elif feat == 'month': df[new_col_name] = dt_col.dt.month
                elif feat == 'day': df[new_col_name] = dt_col.dt.day
                elif feat == 'weekday': df[new_col_name] = dt_col.dt.weekday
                elif feat == 'hour': df[new_col_name] = dt_col.dt.hour
        
        return df
        
    def serialize(self):
        return {
            "op": self.name,
            "column": self.column,
            "features": self.features
        }

class Balance(Operation):
    name = "balance"
    
    def __init__(self, target, strategy='oversample', random_state=42):
        self.target = target
        self.strategy = strategy
        self.random_state = random_state
    
    def apply(self, df):
        from sklearn.utils import resample
        
        if self.strategy == 'oversample' or self.strategy == 'undersample':
            # Identify majority/minority classes automatically (binary simplified)
            # For multi-class, we'd need more complex logic. 
            # Assuming binary for simplicity or just resampling the smaller one.
            counts = df[self.target].value_counts()
            if len(counts) < 2: return df # Nothing to balance
            
            major_class = counts.idxmax()
            minor_class = counts.idxmin()
            
            df_major = df[df[self.target] == major_class]
            df_minor = df[df[self.target] == minor_class]
            
            if self.strategy == 'oversample':
                # Upsample minority
                df_minor_upsampled = resample(df_minor, 
                                              replace=True,     # sample with replacement
                                              n_samples=len(df_major),    # to match majority class
                                              random_state=self.random_state) 
                return pd.concat([df_major, df_minor_upsampled])
            
            elif self.strategy == 'undersample':
                # Downsample majority
                df_major_downsampled = resample(df_major, 
                                                replace=False,    # sample without replacement
                                                n_samples=len(df_minor),  # to match minority class
                                                random_state=self.random_state) 
                return pd.concat([df_major_downsampled, df_minor])

        elif self.strategy == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                sm = SMOTE(random_state=self.random_state)
                X = df.drop(columns=[self.target])
                y = df[self.target]
                
                # Check for NaNs because SMOTE fails with them
                if X.isnull().sum().sum() > 0:
                     print("Warning: Missing values found. SMOTE might fail. Ensure imputation is done before balancing.")
                     
                X_res, y_res = sm.fit_resample(X, y)
                # Recombine
                df_res = pd.concat([X_res, y_res], axis=1)
                return df_res
            except ImportError:
                print("Warning: imblearn not installed. Falling back to random oversampling.")
                # Fallback recursion
                return Balance(self.target, strategy='oversample', random_state=self.random_state).apply(df)
            except Exception as e:
                print(f"SMOTE failed ({e}). Falling back to random oversampling.")
                return Balance(self.target, strategy='oversample', random_state=self.random_state).apply(df)

        return df

    def serialize(self):
        return {
            "op": self.name,
            "target": self.target,
            "strategy": self.strategy
        }

class GenericPandasOp(Operation):
    name = 'generic_pandas'

    def __init__(self, method_name, *args, **kwargs):
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
    
    def apply(self, df):
        # Dynamically call the method on the dataframe
        method = getattr(df, self.method_name)
        return method(*self.args, **self.kwargs)
    
    def serialize(self):
        return {
            "op": self.name,
            "method": self.method_name,
            "args": self.args,
            "kwargs": self.kwargs
        }

# --- Aliases for Backward Compatibility ---
class ImputeMean(Impute):
    name = "impute_mean"
    def __init__(self, column, mean_value):
        # Just delegate with 'mean' strategy
        super().__init__(column, strategy='mean')

class Normalization(Scale):
    name = "normalize"
    def __init__(self, column, normalizer):
        # This one is tricky because original Normalizer took a scaler object.
        # We can map back if it's a known type or just accept it as 'method=standard' if generic
        method = 'standard'
        if 'MinMax' in str(type(normalizer)): method='minmax'
        super().__init__(column, method=method)
        self.scaler = normalizer # Use the passed one

class Ohe(Encode):
    name = "one_hot_encode"
    def __init__(self, column, categories):
         # Old Ohe didn't check categories in init, just stored them.
         # The new one generates them. We can ignore categories param for now as it was result-based.
        super().__init__(column, method='onehot')
