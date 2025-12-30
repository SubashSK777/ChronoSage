import os
import yaml
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def setup_logging(log_path: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    Path(log_path).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_path, f"chronosage_{timestamp}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ChronoSage")
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """Validate DataFrame structure and content."""
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logging.warning(f"Missing required columns: {missing_cols}")
            return False
    
    return True

def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Automatically detect column types in DataFrame."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

def normalize_data(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Normalize numerical columns using min-max scaling."""
    df_normalized = df.copy()
    cols_to_normalize = columns if columns else df.select_dtypes(include=[np.number]).columns
    
    for col in cols_to_normalize:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df_normalized

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate basic statistics for DataFrame."""
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numerical_stats': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    return stats

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format timestamp for file names and logs."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")

def save_results(data: Union[pd.DataFrame, Dict], output_path: str, format: str = "csv") -> bool:
    """Save results to file."""
    try:
        ensure_directory(os.path.dirname(output_path))
        
        if isinstance(data, pd.DataFrame):
            if format == "csv":
                data.to_csv(output_path, index=False)
            elif format == "json":
                data.to_json(output_path, orient='records', indent=2)
            elif format == "excel":
                data.to_excel(output_path, index=False)
        elif isinstance(data, dict):
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logging.info(f"Results saved to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        return False

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load data from file (CSV or JSON)."""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        else:
            logging.error(f"Unsupported file format: {file_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None

class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logging.info(f"{self.name} started at {self.start_time}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logging.info(f"{self.name} completed in {duration:.2f} seconds")
        return False

def retry_on_failure(max_attempts: int = 3, delay: int = 5):
    """Decorator for retrying functions on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_attempts - 1:
                        logging.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logging.error(f"All {max_attempts} attempts failed for {func.__name__}")
                        raise
            return None
        return wrapper
    return decorator
