from datetime import datetime
import json
import pandas as pd
from .operations import *
from .utils import hash_data
from sklearn.preprocessing import StandardScaler

class AuditTrialRecorder:
    def __init__(self, dataframe, name="experiment"):
        self.name = name
        self.original_df = dataframe.copy()
        self.current_df = dataframe.copy()
        self.operations = []
        self.hashes = []

        self._record(
            LoadData(
                shape=dataframe.shape,
                columns=list(dataframe.columns),
                dtypes={c: str(t) for c, t in dataframe.dtypes.items()}
            ))
        
    def _record(self, operation: Operation):
        self.operations.append(operation) 
        self.hashes.append(hash_data(self.current_df))

    # --- Preprocessing Methods ---

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

    # New Comprehensive Impute
    def impute(self, column, strategy='mean', fill_value=None, method=None):
        """
        Impute missing values.
        strategy: 'mean', 'median', 'mode', 'constant'
        method: 'ffill', 'bfill'
        """
        op = Impute(column, strategy, fill_value, method)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    # Legacy support
    def impute_mean(self, column):
        return self.impute(column, strategy='mean')

    # Comprehensive Scale
    def scale(self, column, method='standard'):
        """
        Scale features.
        method: 'standard', 'minmax', 'robust', 'maxabs'
        """
        op = Scale(column, method)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self
        
    def normalize_column(self, column):
        return self.scale(column, method='standard')

    # Comprehensive Encode
    def encode(self, column, method='onehot', target_col=None):
        """
        Encode categorical features.
        method: 'onehot', 'label', 'target'
        """
        op = Encode(column, method, target_col)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    def one_hot_encode(self, column):
        return self.encode(column, method='onehot')

    # Transform (Math)
    def transform(self, column, func='log'):
        """
        Apply mathematical transformation.
        func: 'log', 'sqrt', 'cbrt', 'square'
        """
        op = Transform(column, func)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    # Discretize (Binning)
    def bin_numeric(self, column, bins=5, strategy='quantile', labels=None):
        """
        Bin continuous variables.
        strategy: 'quantile' (equal freq), 'uniform' (equal width)
        """
        op = Discretize(column, bins, strategy, labels)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    # Date Extract
    def extract_date_features(self, column, features=['year', 'month', 'day', 'weekday']):
        op = DateExtract(column, features)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    # Class Balance
    def balance_classes(self, target, strategy='oversample', random_state=42):
        """
        Balance classes in the dataset.
        strategy: 'oversample', 'undersample', 'smote'
        """
        op = Balance(target, strategy, random_state)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    # --- Generic Methods ---

    def track_pandas(self, method_name, *args, **kwargs):
        """
        Apply any pandas method (that returns a DataFrame) and record it.
        Example: auditor.track_pandas('dropna', subset=['age'])
        """
        op = GenericPandasOp(method_name, *args, **kwargs)
        self.current_df = op.apply(self.current_df)
        self._record(op)
        return self

    # --- Replay & Export ---

    def replay(self):
        df = self.original_df.copy()
        for op in self.operations[1:]: # Skip LoadData
            df = op.apply(df)
        return df

    def verify_reproducibility(self):
        try:
            replayed_df = self.replay()
            # If balancing happened, row order might shuffle in some implementations, 
            # but our implementation of concat is deterministic.
            # Reset index might be needed if index became non-unique and messy.
            
            # Simple hash check
            return hash_data(replayed_df) == hash_data(self.current_df)
        except Exception as e:
            print(f"Reproduction failed with error: {e}")
            return False
    
    def export_audit_trail(self, filename=None, output_dir=None, visualize=True):
        import os
        from .visualizer import generate_visualization
        
        if filename is None:
            filename = f"{self.name}_audit.json"
        
        # Determine the output directory
        if output_dir is None:
            # Find the ml_audit package directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to ProjectsML directory
            project_dir = os.path.dirname(package_dir)
            # Set audit_trails as the default output directory
            output_dir = os.path.join(project_dir, 'audit_trails')
            viz_dir = os.path.join(project_dir, 'visualizations')
        else:
            viz_dir = output_dir # If custom dir, put html there too
        
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
             os.makedirs(viz_dir, exist_ok=True)
        
        # Build the full path
        if not os.path.isabs(filename):
            filepath = os.path.join(output_dir, filename)
        else:
            filepath = filename
        
        trail = {
            "experiment": self.name,
            "created": datetime.now().isoformat(),
            "operations": [op.serialize() for op in self.operations],
            "final_hash": hash_data(self.current_df),
            "final_shape": self.current_df.shape,
            "final_columns": list(self.current_df.columns)
        }
        with open(filepath, "w") as f:
            json.dump(trail, f, indent=2)
        
        print(f"Audit trail saved to: {filepath}")
        
        if visualize:
            # Determine html path
            # If default logic used: 
            # json -> audit_trails/name.json
            # html -> visualizations/name.html
            
            basename = os.path.basename(filepath)
            html_name = basename.replace('.json', '.html')
            html_path = os.path.join(viz_dir, html_name)
            
            generate_visualization(filepath, html_path)
            
        return trail

