import hashlib
import pandas as pd

def hash_data(df: pd.DataFrame) -> str:
    """Computes a stable SHA256 hash of a pandas DataFrame."""
    try:
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()
    except Exception:
        # Fallback for complex types or if hash_pandas_object fails
        return hashlib.sha256(str(df.values).encode('utf-8')).hexdigest()
