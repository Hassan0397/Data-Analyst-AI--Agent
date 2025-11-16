import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional
import io
import warnings
import re
from datetime import datetime

class DataProcessor:
    """Enhanced data processing and cleaning utilities"""
    
    def __init__(self):
        self.cleaning_history = []
        self.operations_queue = []
    
    def load_data(self, file) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            if hasattr(file, 'type'):
                file_type = file.type
                file_name = file.name
            else:
                file_type = 'unknown'
                file_name = str(file)
            
            df = None
            
            # Enhanced file type detection
            if file_type == "text/csv" or file_name.endswith('.csv'):
                df = pd.read_csv(file, encoding_errors='ignore', low_memory=False)
            elif file_type in ["application/vnd.ms-excel", 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or \
                 file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file)
            elif file_type == "application/json" or file_name.endswith('.json'):
                df = pd.read_json(file)
            elif file_type == "application/octet-stream" or file_name.endswith('.parquet'):
                df = pd.read_parquet(file)
            else:
                # Try all formats with error handling
                try:
                    df = pd.read_csv(file, encoding_errors='ignore')
                except:
                    try:
                        df = pd.read_excel(file)
                    except:
                        try:
                            df = pd.read_json(file)
                        except:
                            raise ValueError(f"Unsupported file format: {file_name}")
            
            # Enhanced initial cleaning
            df = self.clean_column_names(df)
            df = self.optimize_data_types(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading file {file_name}: {str(e)}")
            raise
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean column names for consistency with enhanced cleaning"""
        def clean_name(name):
            name = str(name).strip().lower()
            # Replace multiple spaces/special chars with single underscore
            name = re.sub(r'[^\w]', '_', name)
            name = re.sub(r'_+', '_', name)
            name = name.strip('_')
            # Ensure name is valid and not empty
            if not name or name.isdigit():
                name = f'column_{name}' if name else 'unnamed_column'
            return name
        
        df.columns = [clean_name(col) for col in df.columns]
        return df
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency"""
        result_df = df.copy()
        
        for col in result_df.columns:
            col_type = result_df[col].dtype
            
            # Optimize numeric columns
            if np.issubdtype(col_type, np.number):
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                result_df[col] = self._optimize_numeric_column(result_df[col])
            
            # Optimize string columns
            elif col_type == 'object':
                # Convert to categorical if low cardinality
                unique_ratio = result_df[col].nunique() / len(result_df)
                if unique_ratio < 0.5 and result_df[col].nunique() < 1000:
                    result_df[col] = result_df[col].astype('category')
                else:
                    result_df[col] = result_df[col].astype('string')
        
        return result_df
    
    def _optimize_numeric_column(self, series: pd.Series) -> pd.Series:
        """Optimize numeric column data type"""
        if series.isna().all():
            return series
        
        # Try to convert to integer if possible
        if (series.dropna() % 1 == 0).all():
            min_val, max_val = series.min(), series.max()
            if min_val > 0:
                if max_val < 255:
                    return series.astype('uint8')
                elif max_val < 65535:
                    return series.astype('uint16')
                elif max_val < 4294967295:
                    return series.astype('uint32')
            else:
                if min_val > -128 and max_val < 127:
                    return series.astype('int8')
                elif min_val > -32768 and max_val < 32767:
                    return series.astype('int16')
                elif min_val > -2147483648 and max_val < 2147483647:
                    return series.astype('int32')
        
        # Use float32 if possible
        if series.dtype == 'float64':
            try:
                return series.astype('float32')
            except:
                pass
        
        return series
    
    def add_operation(self, operation_type: str, **kwargs):
        """Add operation to queue with validation"""
        self.operations_queue.append({
            'type': operation_type,
            'kwargs': kwargs,
            'description': self._generate_operation_description(operation_type, kwargs),
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_operation_description(self, operation_type: str, kwargs: Dict) -> str:
        """Generate human-readable operation description"""
        descriptions = {
            'dropna': f"Remove rows with missing values in {kwargs.get('subset', 'all columns')}",
            'fillna': f"Fill missing values using {kwargs.get('method', 'specified value')}",
            'drop_duplicates': f"Remove duplicate rows based on {kwargs.get('subset', 'all columns')}",
            'astype': f"Convert {kwargs.get('columns', 'specified columns')} to {kwargs.get('dtype', 'target type')}",
            'remove_outliers': f"Remove outliers from {kwargs.get('columns', 'specified columns')}",
            'cap_outliers': f"Cap outliers in {kwargs.get('columns', 'specified columns')}",
            'log_transform': f"Apply log transformation to {kwargs.get('columns', 'specified columns')}",
            'sqrt_transform': f"Apply square root transformation to {kwargs.get('columns', 'specified columns')}",
            'create_feature': f"Create new feature: {kwargs.get('new_column', 'new_column')}",
            'extract_date_feature': f"Extract {kwargs.get('feature', 'date feature')} from {kwargs.get('date_column', 'date column')}",
            'interpolate': f"Interpolate missing values in {kwargs.get('columns', 'specified columns')}",
            'bin_equal_width': f"Create equal-width bins for {kwargs.get('columns', 'specified columns')}",
            'bin_equal_freq': f"Create equal-frequency bins for {kwargs.get('columns', 'specified columns')}",
            'mark_outliers': f"Mark outliers in {kwargs.get('columns', 'specified columns')}"
        }
        
        return descriptions.get(operation_type, f"Operation: {operation_type}")
    
    def apply_all_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all queued operations with enhanced error handling"""
        result_df = df.copy()
        
        for operation in self.operations_queue:
            try:
                result_df = self._apply_single_operation(result_df, operation)
                self.cleaning_history.append({
                    **operation,
                    'rows_affected': len(result_df),
                    'columns_affected': len(result_df.columns)
                })
            except Exception as e:
                print(f"Error applying operation {operation}: {str(e)}")
                raise
        
        self.operations_queue.clear()
        return result_df
    
    def _apply_single_operation(self, df: pd.DataFrame, operation: Dict) -> pd.DataFrame:
        """Apply a single operation to dataframe"""
        op_type = operation['type']
        kwargs = operation['kwargs']
        
        if op_type == 'dropna':
            return df.dropna(**kwargs)
        elif op_type == 'fillna':
            return self._fillna_advanced(df, **kwargs)
        elif op_type == 'drop_duplicates':
            return df.drop_duplicates(**kwargs)
        elif op_type == 'astype':
            return self._convert_dtypes_advanced(df, **kwargs)
        elif op_type == 'remove_outliers':
            return self._remove_outliers_advanced(df, **kwargs)
        elif op_type == 'cap_outliers':
            return self._cap_outliers_advanced(df, **kwargs)
        elif op_type == 'log_transform':
            return self._log_transform(df, **kwargs)
        elif op_type == 'sqrt_transform':
            return self._sqrt_transform(df, **kwargs)
        elif op_type == 'create_feature':
            return self._create_feature_safe(df, **kwargs)
        elif op_type == 'extract_date_feature':
            return self._extract_date_feature_advanced(df, **kwargs)
        elif op_type == 'interpolate':
            return self._interpolate_advanced(df, **kwargs)
        elif op_type == 'bin_equal_width':
            return self._bin_equal_width(df, **kwargs)
        elif op_type == 'bin_equal_freq':
            return self._bin_equal_freq(df, **kwargs)
        elif op_type == 'mark_outliers':
            return self._mark_outliers_advanced(df, **kwargs)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    def _fillna_advanced(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Advanced missing value imputation"""
        result_df = df.copy()
        columns = kwargs.get('columns', df.columns)
        strategy = kwargs.get('strategy', 'constant')
        
        for col in columns:
            if col not in result_df.columns:
                continue
                
            if strategy == 'mean':
                result_df[col] = result_df[col].fillna(result_df[col].mean())
            elif strategy == 'median':
                result_df[col] = result_df[col].fillna(result_df[col].median())
            elif strategy == 'mode':
                result_df[col] = result_df[col].fillna(result_df[col].mode()[0] if not result_df[col].mode().empty else None)
            elif strategy == 'forward_fill':
                result_df[col] = result_df[col].fillna(method='ffill')
            elif strategy == 'backward_fill':
                result_df[col] = result_df[col].fillna(method='bfill')
            elif strategy == 'interpolate':
                result_df[col] = result_df[col].interpolate()
            elif 'value' in kwargs:
                result_df[col] = result_df[col].fillna(kwargs['value'])
        
        return result_df
    
    def _convert_dtypes_advanced(self, df: pd.DataFrame, dtype: str, columns: List[str]) -> pd.DataFrame:
        """Convert data types with enhanced handling"""
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                continue
                
            try:
                if dtype == 'numeric':
                    result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
                elif dtype == 'datetime':
                    result_df[col] = self._smart_date_conversion(result_df[col])
                elif dtype == 'category':
                    result_df[col] = result_df[col].astype('category')
                elif dtype == 'string':
                    result_df[col] = result_df[col].astype('string')
                elif dtype == 'boolean':
                    result_df[col] = self._convert_to_boolean(result_df[col])
            except Exception as e:
                print(f"Warning: Could not convert {col} to {dtype}: {str(e)}")
        
        return result_df
    
    def _smart_date_conversion(self, series: pd.Series) -> pd.Series:
        """Smart date conversion with multiple format attempts"""
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y.%m.%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d.%m.%Y',
            '%Y%m%d', '%d%m%Y', '%m%d%Y', '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M',
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f'
        ]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # First try pandas auto-detection
            converted = pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            
            # If that fails, try specific formats
            if converted.isna().all():
                for fmt in date_formats:
                    converted = pd.to_datetime(series, errors='coerce', format=fmt)
                    if not converted.isna().all():
                        break
            
            return converted
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert various formats to boolean"""
        true_values = ['true', '1', 'yes', 'y', 't']
        false_values = ['false', '0', 'no', 'n', 'f']
        
        def convert_value(x):
            if pd.isna(x):
                return False
            x_str = str(x).lower().strip()
            if x_str in true_values:
                return True
            elif x_str in false_values:
                return False
            else:
                try:
                    return bool(float(x))
                except:
                    return False
        
        return series.apply(convert_value)
    
    def _remove_outliers_advanced(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers using multiple methods"""
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns or not np.issubdtype(result_df[col].dtype, np.number):
                continue
                
            if method == 'iqr':
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                result_df = result_df[
                    (result_df[col] >= lower_bound) & 
                    (result_df[col] <= upper_bound)
                ]
        
        return result_df
    
    def _cap_outliers_advanced(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Cap outliers using multiple methods"""
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns or not np.issubdtype(result_df[col].dtype, np.number):
                continue
                
            if method == 'iqr':
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                result_df[col] = np.where(result_df[col] < lower_bound, lower_bound, result_df[col])
                result_df[col] = np.where(result_df[col] > upper_bound, upper_bound, result_df[col])
        
        return result_df
    
    def _log_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation with enhanced handling"""
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
                min_val = result_df[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    result_df[col] = np.log1p(result_df[col] + shift)
                else:
                    result_df[col] = np.log1p(result_df[col])
        
        return result_df
    
    def _sqrt_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply square root transformation"""
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
                min_val = result_df[col].min()
                if min_val < 0:
                    shift = abs(min_val)
                    result_df[col] = np.sqrt(result_df[col] + shift)
                else:
                    result_df[col] = np.sqrt(result_df[col])
        
        return result_df
    
    def _create_feature_safe(self, df: pd.DataFrame, expression: str, new_column: str) -> pd.DataFrame:
        """Create new feature using safe expression evaluation"""
        result_df = df.copy()
        
        try:
            # Safe evaluation with limited namespace
            safe_dict = {
                'df': result_df,
                'pd': pd,
                'np': np,
                'sqrt': np.sqrt,
                'log': np.log,
                'exp': np.exp,
                'abs': np.abs,
                'sum': np.sum,
                'mean': np.mean,
                'std': np.std
            }
            
            # Create the new column using eval with safe environment
            result_df[new_column] = pd.eval(expression, local_dict=safe_dict)
        except Exception as e:
            print(f"Error creating feature {new_column}: {str(e)}")
            # Fallback: try basic operations
            try:
                if '+' in expression:
                    cols = expression.split('+')
                    result_df[new_column] = result_df[cols[0].strip()] + result_df[cols[1].strip()]
                elif '-' in expression:
                    cols = expression.split('-')
                    result_df[new_column] = result_df[cols[0].strip()] - result_df[cols[1].strip()]
                elif '*' in expression:
                    cols = expression.split('*')
                    result_df[new_column] = result_df[cols[0].strip()] * result_df[cols[1].strip()]
                elif '/' in expression:
                    cols = expression.split('/')
                    result_df[new_column] = result_df[cols[0].strip()] / result_df[cols[1].strip()]
            except:
                result_df[new_column] = np.nan
        
        return result_df
    
    def _extract_date_feature_advanced(self, df: pd.DataFrame, date_column: str, feature: str) -> pd.DataFrame:
        """Extract advanced date features"""
        result_df = df.copy()
        
        if date_column not in result_df.columns:
            return result_df
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
            result_df[date_column] = self._smart_date_conversion(result_df[date_column])
        
        valid_dates = result_df[date_column].notna()
        
        date_features_map = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day': lambda x: x.dt.day,
            'dayofweek': lambda x: x.dt.dayofweek,
            'dayofyear': lambda x: x.dt.dayofyear,
            'week': lambda x: x.dt.isocalendar().week,
            'quarter': lambda x: x.dt.quarter,
            'is_weekend': lambda x: x.dt.dayofweek >= 5,
            'is_month_start': lambda x: x.dt.is_month_start,
            'is_month_end': lambda x: x.dt.is_month_end,
            'hour': lambda x: x.dt.hour,
            'minute': lambda x: x.dt.minute,
            'second': lambda x: x.dt.second
        }
        
        if feature in date_features_map:
            new_col_name = f"{date_column}_{feature}"
            result_df[new_col_name] = date_features_map[feature](result_df.loc[valid_dates, date_column])
            
            # Fill NaN values appropriately
            if feature in ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'week', 'quarter', 'hour', 'minute', 'second']:
                result_df[new_col_name] = result_df[new_col_name].fillna(-1)
            else:
                result_df[new_col_name] = result_df[new_col_name].fillna(False)
        
        return result_df
    
    def _interpolate_advanced(self, df: pd.DataFrame, columns: List[str], method: str = 'linear') -> pd.DataFrame:
        """Advanced interpolation with multiple methods"""
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
                result_df[col] = result_df[col].interpolate(method=method, limit_direction='both')
        
        return result_df
    
    def _bin_equal_width(self, df: pd.DataFrame, columns: List[str], bins: int = 10) -> pd.DataFrame:
        """Create equal-width bins"""
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
                result_df[f'{col}_binned'] = pd.cut(result_df[col], bins=bins, duplicates='drop')
        
        return result_df
    
    def _bin_equal_freq(self, df: pd.DataFrame, columns: List[str], bins: int = 10) -> pd.DataFrame:
        """Create equal-frequency bins"""
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
                result_df[f'{col}_binned'] = pd.qcut(result_df[col], q=bins, duplicates='drop')
        
        return result_df
    
    def _mark_outliers_advanced(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """Mark outliers using multiple methods"""
        result_df = df.copy()
        
        for col in columns:
            if col in result_df.columns and np.issubdtype(result_df[col].dtype, np.number):
                if method == 'iqr':
                    Q1 = result_df[col].quantile(0.25)
                    Q3 = result_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    result_df[f'{col}_is_outlier'] = (
                        (result_df[col] < lower_bound) | 
                        (result_df[col] > upper_bound)
                    )
        
        return result_df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).sum(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'basic_stats': df.describe().to_dict() if not numeric_cols.empty else {},
            'categorical_stats': {col: {
                'top_value': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'top_frequency': df[col].value_counts().iloc[0] if not df[col].value_counts().empty else 0
            } for col in categorical_cols} if not categorical_cols.empty else {}
        }
    
    def detect_anomalies(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect anomalies with multiple methods"""
        anomalies = pd.DataFrame()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                col_anomalies = df[
                    (df[col] < lower_bound) | 
                    (df[col] > upper_bound)
                ].copy()
                
                if not col_anomalies.empty:
                    col_anomalies['anomaly_column'] = col
                    col_anomalies['anomaly_type'] = 'outlier'
                    col_anomalies['anomaly_value'] = col_anomalies[col]
                    anomalies = pd.concat([anomalies, col_anomalies])
        
        return anomalies.drop_duplicates().reset_index(drop=True)
    
    def get_cleaning_history(self) -> List[Dict]:
        """Get history of cleaning operations"""
        return self.cleaning_history
    
    def get_operations_queue(self) -> List[Dict]:
        """Get current operations queue"""
        return self.operations_queue
    
    def clear_operations(self):
        """Clear all queued operations"""
        self.operations_queue.clear()
    
    def export_cleaning_template(self) -> Dict:
        """Export cleaning operations as reusable template"""
        return {
            'operations': self.cleaning_history,
            'export_timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }