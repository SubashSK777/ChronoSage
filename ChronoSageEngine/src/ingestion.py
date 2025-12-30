import os
import logging
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from src.utils import (
    load_config, ensure_directory, validate_dataframe, 
    detect_column_types, save_results, load_data, PerformanceTimer
)

logger = logging.getLogger("ChronoSage.Ingestion")

class DataIngestion:
    """Handle automated data ingestion and preprocessing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data ingestion module."""
        self.config = config or load_config()
        self.data_config = self.config.get('data', {})
        self.ingestion_config = self.config.get('ingestion', {})
        
        self.raw_path = self.data_config.get('raw_path', 'data/raw')
        self.processed_path = self.data_config.get('processed_path', 'data/processed')
        
        ensure_directory(self.raw_path)
        ensure_directory(self.processed_path)
        
        logger.info("DataIngestion module initialized")
    
    def ingest_from_file(self, file_path: str) -> Optional[pd.DataFrame]:
        """Ingest data from CSV or JSON file."""
        try:
            with PerformanceTimer(f"Ingesting data from {file_path}"):
                df = load_data(file_path)
                
                if df is not None:
                    logger.info(f"Loaded {len(df)} rows from {file_path}")
                    return df
                else:
                    logger.error(f"Failed to load data from {file_path}")
                    return None
        except Exception as e:
            logger.error(f"Error during file ingestion: {e}")
            return None
    
    def ingest_from_api(self, url: str, params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """Ingest data from REST API endpoint."""
        try:
            with PerformanceTimer(f"Fetching data from API: {url}"):
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    logger.error("Unexpected API response format")
                    return None
                
                logger.info(f"Fetched {len(df)} rows from API")
                return df
        except Exception as e:
            logger.error(f"Error fetching data from API: {e}")
            return None
    
    def preprocess(self, df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
        """Preprocess DataFrame with cleaning and transformation."""
        try:
            with PerformanceTimer("Preprocessing data"):
                df_processed = df.copy()
                
                logger.info(f"Original data shape: {df_processed.shape}")
                
                df_processed = self._handle_missing_values(df_processed)
                df_processed = self._handle_duplicates(df_processed)
                df_processed = self._detect_and_convert_types(df_processed)
                df_processed = self._handle_outliers(df_processed)
                
                logger.info(f"Processed data shape: {df_processed.shape}")
                
                if save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.processed_path, f"processed_data_{timestamp}.csv")
                    save_results(df_processed, output_path, format="csv")
                
                return df_processed
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on configuration."""
        missing_strategy = self.ingestion_config.get('handle_missing', 'drop')
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")
            
            if missing_strategy == 'drop':
                df = df.dropna()
                logger.info("Dropped rows with missing values")
            elif missing_strategy == 'fill_mean':
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
                logger.info("Filled missing numerical values with mean")
            elif missing_strategy == 'fill_median':
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
                logger.info("Filled missing numerical values with median")
            elif missing_strategy == 'fill_mode':
                for col in df.columns:
                    df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col], inplace=True)
                logger.info("Filled missing values with mode")
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Found {duplicates} duplicate rows")
            df = df.drop_duplicates()
            logger.info("Removed duplicate rows")
        
        return df
    
    def _detect_and_convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert column types."""
        if self.ingestion_config.get('auto_detect_types', True):
            column_types = detect_column_types(df)
            logger.info(f"Detected column types: {column_types}")
            
            date_cols = self.ingestion_config.get('date_columns', [])
            for col in date_cols:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        logger.info(f"Converted {col} to datetime")
                    except Exception as e:
                        logger.warning(f"Could not convert {col} to datetime: {e}")
            
            categorical_cols = self.ingestion_config.get('categorical_columns', [])
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
                    logger.info(f"Converted {col} to category")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Detect and handle outliers using IQR method."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        outliers_removed = 0
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outliers_removed += outliers
        
        if outliers_removed > 0:
            logger.info(f"Detected {outliers_removed} potential outliers across numerical columns")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features from existing data."""
        df_features = df.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) >= 2:
            for i, col1 in enumerate(numerical_cols[:3]):
                for col2 in numerical_cols[i+1:4]:
                    interaction_name = f"{col1}_x_{col2}"
                    df_features[interaction_name] = df[col1] * df[col2]
            
            logger.info("Created interaction features")
        
        return df_features
    
    def validate_data(self, df: pd.DataFrame, schema: Optional[Dict] = None) -> bool:
        """Validate data against schema or basic checks."""
        if not validate_dataframe(df):
            logger.error("DataFrame validation failed")
            return False
        
        if schema:
            required_cols = schema.get('required_columns', [])
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns: {set(required_cols) - set(df.columns)}")
                return False
            
            min_rows = schema.get('min_rows', 0)
            if len(df) < min_rows:
                logger.error(f"Insufficient rows: {len(df)} < {min_rows}")
                return False
        
        logger.info("Data validation passed")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        return summary
