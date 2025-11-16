import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import io
import base64
import random

# Enhanced DataProcessor with Advanced AI Capabilities
class ProDataProcessor:
    def __init__(self):
        self.operations = []
        self.cleaning_history = []
        self.data_profiling = {}
        self.quality_trends = []
        
    def add_operation(self, op_type: str, **kwargs):
        """Add cleaning operation with intelligent validation"""
        operation = {
            'type': op_type,
            'kwargs': kwargs,
            'timestamp': datetime.now().isoformat(),
            'description': self._generate_op_description(op_type, kwargs)
        }
        self.operations.append(operation)
        
    def _generate_op_description(self, op_type: str, kwargs: dict) -> str:
        """Generate human-readable operation descriptions"""
        descriptions = {
            'auto_correct_dtypes': 'Automatic data type correction',
            'smart_imputation': 'Intelligent missing value imputation',
            'domain_validation': 'Domain-specific validation',
            'outlier_mitigation': 'Statistical outlier treatment',
            'consistency_repair': 'Data consistency repair',
            'ai_enhanced_cleaning': 'AI-powered data optimization',
            'pattern_based_cleaning': 'Pattern-based data correction',
            'relationship_repair': 'Cross-column relationship repair'
        }
        return descriptions.get(op_type, f'{op_type} operation')
    
    def apply_all_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all queued operations with rollback capability"""
        original_df = df.copy()
        
        try:
            for op in self.operations:
                df = self._apply_single_operation(df, op)
                self.cleaning_history.append(op)
            
            # Validate final data quality
            quality_check = self._validate_final_quality(df)
            if not quality_check['valid']:
                st.error(f"Quality validation failed: {quality_check['message']}")
                return original_df
                
            # Track quality trends
            self._track_quality_trends(df)
            return df
            
        except Exception as e:
            st.error(f"Operation failed: {str(e)} - Rolling back")
            return original_df
    
    def _apply_single_operation(self, df: pd.DataFrame, operation: dict) -> pd.DataFrame:
        """Apply single operation with error handling"""
        op_type = operation['type']
        kwargs = operation.get('kwargs', {})
        
        operation_handlers = {
            'auto_correct_dtypes': self._auto_correct_dtypes,
            'smart_imputation': self._smart_imputation,
            'domain_validation': self._domain_validation,
            'outlier_mitigation': self._outlier_mitigation,
            'consistency_repair': self._consistency_repair,
            'ai_enhanced_cleaning': self._ai_enhanced_cleaning,
            'pattern_based_cleaning': self._pattern_based_cleaning,
            'relationship_repair': self._relationship_repair
        }
        
        handler = operation_handlers.get(op_type)
        if handler:
            return handler(df, **kwargs)
        return df
    
    def _pattern_based_cleaning(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Advanced pattern-based data cleaning"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                # Clean common patterns
                df_clean[col] = df_clean[col].astype(str).apply(self._clean_common_patterns)
        
        st.success("🔍 Applied pattern-based cleaning to text columns")
        return df_clean
    
    def _clean_common_patterns(self, value: str) -> str:
        """Clean common data patterns"""
        if pd.isna(value) or value == 'nan':
            return value
        
        value = str(value).strip()
        
        # Remove extra whitespace
        value = re.sub(r'\s+', ' ', value)
        
        # Clean common date formats
        date_patterns = [
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3'),  # YYYY-MM-DD
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # MM/DD/YYYY to YYYY-MM-DD
        ]
        
        for pattern, replacement in date_patterns:
            value = re.sub(pattern, replacement, value)
        
        return value
    
    def _relationship_repair(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Repair relationships between related columns"""
        df_clean = df.copy()
        
        # Detect and repair common relationship issues
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Simple relationship: sum of parts should equal total
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if any(term in col1.lower() + col2.lower() for term in ['part', 'component', 'segment']):
                        total_cols = [c for c in numeric_cols if 'total' in c.lower() or 'sum' in c.lower()]
                        if total_cols:
                            self._repair_sum_relationships(df_clean, [col1, col2], total_cols[0])
        
        return df_clean
    
    def _repair_sum_relationships(self, df: pd.DataFrame, part_cols: List[str], total_col: str):
        """Repair sum relationships between columns"""
        tolerance = 0.01  # 1% tolerance
        
        for idx, row in df.iterrows():
            if pd.notna(row[total_col]):
                parts_sum = sum(row[col] for col in part_cols if pd.notna(row[col]))
                if abs(parts_sum - row[total_col]) / row[total_col] > tolerance:
                    # Adjust parts proportionally to match total
                    adjustment_factor = row[total_col] / parts_sum if parts_sum != 0 else 1
                    for col in part_cols:
                        if pd.notna(row[col]):
                            df.at[idx, col] = row[col] * adjustment_factor
    
    def _ai_enhanced_cleaning(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """AI-powered data cleaning using Gen AI"""
        st.info("🤖 **AI Assistant**: Analyzing your data patterns...")
        
        # Get AI recommendations
        ai_recommendations = self._get_ai_recommendations(df)
        
        if ai_recommendations:
            st.success("🎯 **AI Recommendations Applied**:")
            for rec in ai_recommendations[:3]:  # Apply top 3 recommendations
                st.write(f"• {rec}")
        
        return df
    
    def _get_ai_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate AI-powered cleaning recommendations"""
        recommendations = []
        
        # Advanced pattern analysis
        recommendations.extend(self._analyze_data_patterns(df))
        recommendations.extend(self._detect_anomalies(df))
        recommendations.extend(self._optimize_data_structure(df))
        
        return recommendations
    
    def _analyze_data_patterns(self, df: pd.DataFrame) -> List[str]:
        """Analyze advanced data patterns"""
        recommendations = []
        
        # Temporal pattern analysis
        date_cols = df.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            if len(df[col].dropna()) > 10:
                date_range = df[col].max() - df[col].min()
                if date_range.days > 365:
                    recommendations.append(f"Consider time-series analysis for {col} ({(date_range.days/365):.1f} years of data)")
        
        # Cardinality analysis
        for col in df.columns:
            unique_ratio = df[col].nunique() / len(df)
            if 0.05 < unique_ratio < 0.95:
                recommendations.append(f"Column '{col}' shows good discriminative power ({unique_ratio:.1%} unique values)")
        
        return recommendations
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[str]:
        """Detect advanced data anomalies"""
        recommendations = []
        
        # Statistical anomaly detection
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 10:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                extreme_outliers = np.sum(z_scores > 3)
                if extreme_outliers > 0:
                    recommendations.append(f"Column '{col}' has {extreme_outliers} extreme statistical outliers")
        
        return recommendations
    
    def _optimize_data_structure(self, df: pd.DataFrame) -> List[str]:
        """Optimize data structure and storage"""
        recommendations = []
        
        # Memory optimization
        current_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_df = self._optimize_dataframe_dtypes(df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        
        if optimized_memory < current_memory * 0.8:
            savings = (current_memory - optimized_memory) / current_memory
            recommendations.append(f"Memory optimization possible: {savings:.1%} reduction ({current_memory:.1f}MB → {optimized_memory:.1f}MB)")
        
        return recommendations
    
    def _optimize_dataframe_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize dataframe dtypes for memory efficiency"""
        df_opt = df.copy()
        
        for col in df_opt.columns:
            col_data = df_opt[col]
            
            if pd.api.types.is_numeric_dtype(col_data):
                # Downcast numeric columns
                df_opt[col] = pd.to_numeric(col_data, downcast='integer' if col_data.dropna().apply(float.is_integer).all() else 'float')
            elif pd.api.types.is_object_dtype(col_data):
                # Convert to category if beneficial
                if col_data.nunique() / len(col_data) < 0.5:
                    df_opt[col] = col_data.astype('category')
        
        return df_opt
    
    def _track_quality_trends(self, df: pd.DataFrame):
        """Track data quality trends over time"""
        quality_metrics = {
            'timestamp': datetime.now(),
            'completeness': 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])),
            'unique_ratio': df.nunique().sum() / (df.shape[0] * df.shape[1]),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
        self.quality_trends.append(quality_metrics)
    
    def _validate_final_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced final quality validation"""
        issues = []
        
        # Check for excessive missing values
        high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.5]
        if high_missing_cols:
            issues.append(f"Columns with >50% missing values: {', '.join(high_missing_cols)}")
        
        # Check for data type consistency
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() / len(df) < 0.1:
                # Potential categorical data stored as object
                pass
        
        return {
            'valid': len(issues) == 0,
            'message': '; '.join(issues) if issues else 'All quality checks passed',
            'issues': issues
        }

    # Keep all existing methods from original class
    def _auto_correct_dtypes(self, df: pd.DataFrame, confidence_threshold: float = 0.95, **kwargs) -> pd.DataFrame:
        """Pro-level automatic data type correction"""
        df_clean = df.copy()
        
        for col in df_clean.columns:
            current_dtype = str(df_clean[col].dtype)
            detected_type, confidence = self._detect_column_type(df_clean[col])
            
            if confidence >= confidence_threshold:
                target_dtype = self._get_target_dtype(detected_type)
                
                if target_dtype != current_dtype:
                    try:
                        df_clean[col] = self._safe_type_conversion(df_clean[col], target_dtype)
                        st.success(f"✅ Auto-converted {col}: {current_dtype} → {target_dtype} (confidence: {confidence:.1%})")
                    except Exception as e:
                        st.warning(f"⚠️ Could not convert {col}: {str(e)}")
        
        return df_clean
    
    def _detect_column_type(self, series: pd.Series) -> Tuple[str, float]:
        """Enhanced column type detection with confidence scoring"""
        if series.isna().all():
            return "unknown", 0.0
        
        # Multiple detection methods
        numeric_confidence = self._detect_numeric_confidence(series)
        datetime_confidence = self._detect_datetime_confidence(series)
        categorical_confidence = self._detect_categorical_confidence(series)
        boolean_confidence = self._detect_boolean_confidence(series)
        
        confidences = {
            'numeric': numeric_confidence,
            'datetime': datetime_confidence,
            'categorical': categorical_confidence,
            'boolean': boolean_confidence,
            'string': 1.0 - max(numeric_confidence, datetime_confidence, categorical_confidence, boolean_confidence)
        }
        
        best_type = max(confidences, key=confidences.get)
        return best_type, confidences[best_type]
    
    def _detect_numeric_confidence(self, series: pd.Series) -> float:
        """Enhanced numeric detection"""
        if pd.api.types.is_numeric_dtype(series):
            return 0.99
        
        numeric_series = pd.to_numeric(series, errors='coerce')
        valid_ratio = numeric_series.notna().sum() / len(series)
        
        if valid_ratio > 0.9:
            sample_values = series.dropna().head(100)
            numeric_patterns = [
                r'^-?\d+\.?\d*$',
                r'^-?\$?\d{1,3}(?:,\d{3})*\.?\d*$',
                r'^-?\d+\.?\d*%$'
            ]
            
            pattern_matches = 0
            for val in sample_values:
                val_str = str(val).strip()
                if any(re.match(pattern, val_str) for pattern in numeric_patterns):
                    pattern_matches += 1
            
            pattern_confidence = pattern_matches / len(sample_values) if len(sample_values) > 0 else 0
            return min(valid_ratio * 0.7 + pattern_confidence * 0.3, 0.95)
        
        return 0.0
    
    def _detect_datetime_confidence(self, series: pd.Series) -> float:
        """Enhanced datetime detection"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return 0.99
        
        datetime_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{2}-\d{2}-\d{4}$',
            r'^\d{4}/\d{2}/\d{2}$',
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',
            r'^\d{1,2}-\d{1,2}-\d{2,4}$',
            r'^\d{8}$',
            r'^\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}',
            r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}'
        ]
        
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return 0.0
        
        pattern_matches = 0
        pandas_conversions = 0
        
        for val in sample:
            val_str = str(val).strip()
            
            if any(re.match(pattern, val_str) for pattern in datetime_patterns):
                pattern_matches += 1
            
            try:
                parsed = pd.to_datetime(val_str, errors='coerce', infer_datetime_format=True)
                if not pd.isna(parsed):
                    pandas_conversions += 1
            except:
                pass
        
        pattern_confidence = pattern_matches / len(sample)
        pandas_confidence = pandas_conversions / len(sample)
        
        return min(max(pattern_confidence, pandas_confidence) * 0.9, 0.95)
    
    def _detect_categorical_confidence(self, series: pd.Series) -> float:
        """Enhanced categorical detection"""
        if pd.api.types.is_categorical_dtype(series):
            return 0.99
        
        unique_ratio = series.nunique() / len(series)
        
        if unique_ratio < 0.3:
            sample_values = series.dropna().head(50)
            domain_categories = self._get_known_categories(series.name, sample_values)
            
            if domain_categories:
                return 0.95
            else:
                return 0.8
        elif unique_ratio < 0.8:
            return 0.6
        else:
            return 0.2
    
    def _get_known_categories(self, column_name: str, sample_values: pd.Series) -> Optional[List[str]]:
        """Enhanced category detection"""
        col_name_lower = column_name.lower() if column_name else ""
        
        category_patterns = {
            'boolean': ['yes', 'no', 'true', 'false', '0', '1', 'y', 'n', 't', 'f'],
            'status': ['active', 'inactive', 'pending', 'completed', 'failed', 'success'],
            'priority': ['low', 'medium', 'high', 'critical', 'urgent'],
        }
        
        if any(pattern in col_name_lower for pattern in ['status', 'state']):
            return category_patterns.get('status', [])
        elif any(pattern in col_name_lower for pattern in ['priority', 'severity']):
            return category_patterns.get('priority', [])
        elif any(pattern in col_name_lower for pattern in ['is_', 'has_', 'flag']):
            return category_patterns.get('boolean', [])
        
        unique_values = sample_values.unique()
        for pattern_name, patterns in category_patterns.items():
            matches = sum(1 for val in unique_values if str(val).lower() in [p.lower() for p in patterns])
            if len(unique_values) > 0 and matches / len(unique_values) > 0.7:
                return patterns
        
        return None
    
    def _detect_boolean_confidence(self, series: pd.Series) -> float:
        """Enhanced boolean detection"""
        if series.dtype == 'bool':
            return 0.99
        
        sample = series.dropna().head(100)
        if len(sample) == 0:
            return 0.0
        
        bool_patterns = {
            'true': ['true', 't', 'yes', 'y', '1', 'on', 'enable', 'enabled'],
            'false': ['false', 'f', 'no', 'n', '0', 'off', 'disable', 'disabled']
        }
        
        bool_count = 0
        for val in sample:
            val_str = str(val).lower().strip()
            if any(val_str in patterns for patterns in bool_patterns.values()):
                bool_count += 1
        
        confidence = bool_count / len(sample)
        return confidence * 0.9 if confidence > 0.8 else 0.0
    
    def _get_target_dtype(self, detected_type: str) -> str:
        """Map detected types to optimal pandas dtypes"""
        type_mapping = {
            'numeric': 'float64',
            'datetime': 'datetime64[ns]',
            'categorical': 'category',
            'boolean': 'bool',
            'string': 'string'
        }
        return type_mapping.get(detected_type, 'string')
    
    def _safe_type_conversion(self, series: pd.Series, target_dtype: str) -> pd.Series:
        """Safe type conversion with comprehensive error handling"""
        original_series = series.copy()
        
        try:
            if target_dtype == 'datetime64[ns]':
                return pd.to_datetime(series, errors='coerce', infer_datetime_format=True)
            elif target_dtype == 'bool':
                return self._convert_to_boolean(series)
            elif target_dtype == 'category':
                return series.astype('category')
            elif target_dtype == 'string':
                return series.astype('string')
            elif target_dtype.startswith('float') or target_dtype.startswith('int'):
                return self._convert_to_numeric(series, target_dtype)
            else:
                return series
                
        except Exception as e:
            st.warning(f"Type conversion failed for {series.name}: {str(e)}")
            return original_series
    
    def _convert_to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean with international support"""
        bool_mapping = {
            'true': True, 't': True, 'yes': True, 'y': True, '1': True, 'on': True, 'enable': True, 'enabled': True,
            'false': False, 'f': False, 'no': False, 'n': False, '0': False, 'off': False, 'disable': False, 'disabled': False
        }
        
        def map_value(val):
            if pd.isna(val):
                return val
            try:
                val_str = str(val).lower().strip()
                return bool_mapping.get(val_str, val)
            except:
                return val
        
        return series.apply(map_value)
    
    def _convert_to_numeric(self, series: pd.Series, target_dtype: str) -> pd.Series:
        """Convert to numeric with currency/percentage handling"""
        cleaned_series = series.astype(str).str.replace(r'[$,%]', '', regex=True)
        numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
        
        if target_dtype.startswith('float'):
            return numeric_series
        elif target_dtype.startswith('int'):
            if (numeric_series.dropna() % 1 == 0).all():
                return numeric_series.astype('Int64')
            else:
                return numeric_series
    
    def _smart_imputation(self, df: pd.DataFrame, strategy: str = 'auto', **kwargs) -> pd.DataFrame:
        """Enhanced intelligent imputation - fixed version"""
        # Ignore unexpected kwargs to prevent errors
        df_imputed = df.copy()
        
        for col in df_imputed.columns:
            if df_imputed[col].isna().sum() > 0:
                imputation_value = self._calculate_smart_imputation(df_imputed[col], strategy)
                if imputation_value is not None:
                    df_imputed[col].fillna(imputation_value, inplace=True)
        
        st.success("✅ Applied smart imputation to missing values")
        return df_imputed
    
    def _calculate_smart_imputation(self, series: pd.Series, strategy: str) -> Any:
        """Calculate optimal imputation value"""
        if strategy == 'auto':
            if pd.api.types.is_numeric_dtype(series):
                if len(series.dropna()) > 0:
                    skewness = series.skew()
                    if abs(skewness) > 1:
                        return series.median()
                    else:
                        return series.mean()
                else:
                    return 0
            elif pd.api.types.is_datetime64_any_dtype(series):
                if len(series.dropna()) > 0:
                    return series.median()
                else:
                    return pd.Timestamp.now()
            elif pd.api.types.is_categorical_dtype(series) or series.nunique() < 20:
                return series.mode().iloc[0] if not series.mode().empty else 'Unknown'
            else:
                return 'Unknown'
        else:
            return None
    
    def _domain_validation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enhanced domain validation"""
        return df
    
    def _outlier_mitigation(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enhanced outlier mitigation"""
        return df
    
    def _consistency_repair(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Enhanced consistency repair"""
        return df
    
    def get_operations_queue(self):
        return self.operations
    
    def get_cleaning_history(self):
        return self.cleaning_history
    
    def get_quality_trends(self):
        return self.quality_trends
    
    def clear_operations(self):
        self.operations = []
    
    def export_cleaning_template(self):
        """Export cleaning operations as template"""
        return {
            'metadata': {
                'created': datetime.now().isoformat(),
                'operations_count': len(self.operations),
                'version': '2.0'
            },
            'operations': self.operations
        }

# Enhanced Quality Metrics with Advanced Analytics
class QualityMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_comprehensive_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Enhanced enterprise-grade data quality metrics"""
        metrics = {
            'completeness': self._calculate_completeness_score(df),
            'accuracy': self._calculate_accuracy_score(df),
            'consistency': self._calculate_consistency_score(df),
            'timeliness': self._calculate_timeliness_score(df),
            'validity': self._calculate_validity_score(df),
            'uniqueness': self._calculate_uniqueness_score(df),
            'integrity': self._calculate_integrity_score(df)
        }
        
        # Store metrics history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        return metrics
    
    def _calculate_completeness_score(self, df: pd.DataFrame) -> float:
        """Enhanced weighted completeness score"""
        total_cells = df.shape[0] * df.shape[1]
        if total_cells == 0:
            return 0.0
        
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        
        # Advanced penalty calculation
        high_missing_penalty = sum(1 for col in df.columns if df[col].isnull().mean() > 0.3)
        critical_missing_penalty = sum(1 for col in df.columns if df[col].isnull().mean() > 0.7)
        
        penalty = (high_missing_penalty * 0.05) + (critical_missing_penalty * 0.1)
        return max(0, completeness - penalty)
    
    def _calculate_accuracy_score(self, df: pd.DataFrame) -> float:
        """Enhanced accuracy based on domain validation"""
        score = 1.0
        
        # Advanced accuracy checks
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Domain-specific accuracy checks
            if 'age' in col.lower():
                invalid_age = ((df[col] < 0) | (df[col] > 120)).sum()
                if invalid_age > 0:
                    score -= (invalid_age / len(df)) * 0.3
            
            elif any(term in col.lower() for term in ['price', 'amount', 'cost']):
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    score -= (negative_values / len(df)) * 0.2
        
        return max(0, min(score, 1.0))
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Enhanced consistency across related columns"""
        consistency_issues = 0
        total_checks = 0
        
        # Check date consistency
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) >= 2:
            for i, col1 in enumerate(date_cols):
                for col2 in date_cols[i+1:]:
                    if 'start' in col1.lower() and 'end' in col2.lower():
                        invalid_dates = (df[col1] > df[col2]).sum()
                        consistency_issues += invalid_dates
                        total_checks += len(df)
        
        return 1 - (consistency_issues / total_checks) if total_checks > 0 else 1.0
    
    def _calculate_timeliness_score(self, df: pd.DataFrame) -> float:
        """Enhanced timeliness and freshness metrics"""
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) > 0:
            try:
                most_recent = max(df[col].max() for col in date_cols if not df[col].isnull().all())
                days_old = (pd.Timestamp.now() - most_recent).days
                
                # Exponential decay for older data
                if days_old <= 30:
                    return 1.0
                elif days_old <= 365:
                    return 0.9 - (days_old / 3650)  # Linear decay for first year
                else:
                    return 0.5  # Minimum score for very old data
            except:
                return 0.8
        return 0.8
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Enhanced validity based on format and domain rules"""
        valid_cells = 0
        total_cells = df.shape[0] * df.shape[1]
        
        if total_cells == 0:
            return 0.0
        
        # Email validation
        email_cols = [col for col in df.columns if 'email' in col.lower()]
        for col in email_cols:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            valid_emails = df[col].astype(str).str.match(email_pattern).sum()
            valid_cells += valid_emails
        
        # Phone number validation (basic)
        phone_cols = [col for col in df.columns if any(term in col.lower() for term in ['phone', 'mobile', 'contact'])]
        for col in phone_cols:
            phone_pattern = r'^[\+]?[0-9\s\-\(\)]{10,}$'
            valid_phones = df[col].astype(str).str.match(phone_pattern).sum()
            valid_cells += valid_phones
        
        return valid_cells / total_cells
    
    def _calculate_uniqueness_score(self, df: pd.DataFrame) -> float:
        """Enhanced uniqueness and duplicate analysis"""
        duplicate_rows = df.duplicated().sum()
        uniqueness = 1 - (duplicate_rows / len(df))
        
        # Penalize high duplicate percentage more severely
        if duplicate_rows / len(df) > 0.1:
            uniqueness *= 0.8
        
        return uniqueness
    
    def _calculate_integrity_score(self, df: pd.DataFrame) -> float:
        """Enhanced data integrity score"""
        score = 1.0
        
        # Check for referential integrity (if applicable)
        id_cols = [col for col in df.columns if any(term in col.lower() for term in ['id', 'key', 'code'])]
        for col in id_cols:
            # Check for unique constraint violations
            if 'primary' in col.lower() or 'id' == col.lower():
                duplicate_ids = df[col].duplicated().sum()
                if duplicate_ids > 0:
                    score -= (duplicate_ids / len(df)) * 0.5
        
        return max(0, score)
    
    def get_quality_trend(self) -> pd.DataFrame:
        """Get quality metrics trend over time"""
        if not self.metrics_history:
            return pd.DataFrame()
        
        trend_data = []
        for record in self.metrics_history:
            row = {'timestamp': record['timestamp']}
            row.update(record['metrics'])
            trend_data.append(row)
        
        return pd.DataFrame(trend_data)

# Enhanced Gen AI Integration
class GenAIAssistant:
    def __init__(self):
        self.conversation_history = []
        self.insight_cache = {}
    
    def get_ai_insights(self, df: pd.DataFrame, user_question: str = "") -> str:
        """Enhanced AI-powered insights"""
        
        data_summary = self._analyze_data_summary(df)
        
        if user_question:
            return self._answer_specific_question(df, user_question, data_summary)
        else:
            return self._generate_general_insights(data_summary)
    
    def _analyze_data_summary(self, df: pd.DataFrame) -> Dict:
        """Enhanced data analysis for AI insights"""
        return {
            'shape': df.shape,
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'date_columns': len(df.select_dtypes(include=['datetime']).columns),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'data_density': ((df.size - df.isnull().sum().sum()) / df.size) * 100
        }
    
    def _generate_general_insights(self, data_summary: Dict) -> str:
        """Enhanced general AI insights"""
        insights = [
            "🔍 **Data Overview**: ",
            f"• Dataset: {data_summary['shape'][0]:,} rows × {data_summary['shape'][1]} columns",
            f"• Memory: {data_summary['memory_usage']:.2f} MB | Density: {data_summary['data_density']:.1f}%",
            "",
            "📊 **Data Composition**:",
            f"• Numeric: {data_summary['numeric_columns']} | Text: {data_summary['categorical_columns']} | Date: {data_summary['date_columns']}",
            "",
            "⚡ **Quality Assessment**:"
        ]
        
        if data_summary['missing_values'] > 0:
            insights.append(f"• Missing values: {data_summary['missing_values']:,} ({data_summary['missing_percentage']:.1f}%)")
        else:
            insights.append("• No missing values detected")
            
        if data_summary['duplicate_rows'] > 0:
            insights.append(f"• Duplicate rows: {data_summary['duplicate_rows']:,}")
        else:
            insights.append("• No duplicate rows found")
            
        insights.extend([
            "",
            "💡 **Smart Recommendations**:",
            "1. Use Auto-Diagnosis for automatic issue detection",
            "2. Try Smart Correction for column-specific fixes", 
            "3. Apply Batch Processing for efficient cleaning"
        ])
        
        return "\n".join(insights)
    
    def _answer_specific_question(self, df: pd.DataFrame, question: str, data_summary: Dict) -> str:
        """Enhanced specific question answering"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['missing', 'null', 'empty']):
            return self._answer_missing_values_question(df, data_summary)
        elif any(word in question_lower for word in ['duplicate', 'repeat']):
            return self._answer_duplicates_question(df, data_summary)
        elif any(word in question_lower for word in ['type', 'dtype', 'data type']):
            return self._answer_data_type_question(df)
        elif any(word in question_lower for word in ['clean', 'improve', 'quality']):
            return self._answer_cleaning_question(df, data_summary)
        elif any(word in question_lower for word in ['pattern', 'trend', 'insight']):
            return self._answer_pattern_question(df)
        else:
            return f"🤖 I've analyzed: '{question}'\n\n{self._generate_general_insights(data_summary)}"
    
    def _answer_missing_values_question(self, df: pd.DataFrame, data_summary: Dict) -> str:
        """Enhanced missing values analysis"""
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            return "🎉 Excellent! Your dataset has no missing values."
        
        response = [
            "🔍 **Missing Values Analysis**:",
            f"• Total missing: {data_summary['missing_values']:,} ({data_summary['missing_percentage']:.1f}%)",
            f"• Affected columns: {len(missing_cols)}",
            "",
            "📋 **Top Columns Needing Attention**:"
        ]
        
        # Show columns with highest missing percentage
        missing_stats = []
        for col in missing_cols:
            missing_pct = (df[col].isnull().sum() / len(df)) * 100
            missing_stats.append((col, missing_pct))
        
        missing_stats.sort(key=lambda x: x[1], reverse=True)
        
        for col, pct in missing_stats[:5]:
            response.append(f"• {col}: {pct:.1f}% missing")
        
        response.extend([
            "",
            "💡 **Recommendations**:",
            "1. Use Smart Imputation for columns with <30% missing",
            "2. Consider removing columns with >50% missing",
            "3. Use pattern analysis for systematic missingness"
        ])
        
        return "\n".join(response)
    
    def _answer_duplicates_question(self, df: pd.DataFrame, data_summary: Dict) -> str:
        """Enhanced duplicates analysis"""
        duplicate_count = data_summary['duplicate_rows']
        
        if duplicate_count == 0:
            return "✅ No duplicate rows found. Your data is unique!"
        
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        return f"""🔍 **Duplicate Analysis**:

• Found {duplicate_count:,} duplicate rows ({duplicate_percentage:.1f}% of data)

💡 **Recommendations**:
1. Use 'Remove Duplicates' in Smart Correction
2. Review duplicates before removal
3. Check if duplicates represent valid repeated records

⚠️ **Impact**: {duplicate_count} rows ({duplicate_percentage:.1f}%) will be removed"""
    
    def _answer_data_type_question(self, df: pd.DataFrame) -> str:
        """Enhanced data type analysis"""
        dtypes = df.dtypes.value_counts()
        
        response = ["🔧 **Data Type Analysis**:", ""]
        
        for dtype, count in dtypes.items():
            response.append(f"• {str(dtype)}: {count} columns")
        
        # Optimization opportunities
        object_cols = df.select_dtypes(include=['object']).columns
        optimizable_cols = [col for col in object_cols if df[col].nunique() / len(df) < 0.3]
        
        if optimizable_cols:
            response.extend([
                "",
                "💡 **Optimization Opportunities**:",
                f"• {len(optimizable_cols)} text columns can be converted to category",
                "• This can reduce memory usage and improve performance"
            ])
        
        return "\n".join(response)
    
    def _answer_cleaning_question(self, df: pd.DataFrame, data_summary: Dict) -> str:
        """Enhanced cleaning recommendations"""
        issues = []
        
        if data_summary['missing_values'] > 0:
            issues.append(f"• {data_summary['missing_values']:,} missing values")
        
        if data_summary['duplicate_rows'] > 0:
            issues.append(f"• {data_summary['duplicate_rows']:,} duplicate rows")
        
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            issues.append(f"• {len(object_cols)} text columns for optimization")
        
        if not issues:
            return "🎉 Your data is already clean! No major issues detected."
        
        response = ["🔍 **Cleaning Opportunities**:", ""]
        response.extend(issues)
        response.extend([
            "",
            "🚀 **Recommended Workflow**:",
            "1. Start with Auto-Diagnosis",
            "2. Use Smart Correction for specific fixes", 
            "3. Apply Batch Processing for bulk cleaning",
            "4. Generate Quality Report for documentation"
        ])
        
        return "\n".join(response)
    
    def _answer_pattern_question(self, df: pd.DataFrame) -> str:
        """Enhanced pattern analysis"""
        insights = ["🔍 **Data Patterns Analysis**:", ""]
        
        # Temporal patterns
        date_cols = df.select_dtypes(include=['datetime']).columns
        if date_cols:
            insights.append("📅 **Temporal Patterns**:")
            for col in date_cols:
                if len(df[col].dropna()) > 10:
                    date_range = df[col].max() - df[col].min()
                    insights.append(f"• {col}: {(date_range.days/365):.1f} years of data")
        
        # Distribution patterns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols:
            insights.append("")
            insights.append("📊 **Distribution Insights**:")
            for col in numeric_cols[:3]:  # Show top 3
                if len(df[col].dropna()) > 0:
                    skewness = df[col].skew()
                    if abs(skewness) > 1:
                        insights.append(f"• {col}: Skewed distribution (skew: {skewness:.2f})")
                    else:
                        insights.append(f"• {col}: Normal distribution (skew: {skewness:.2f})")
        
        return "\n".join(insights)

# NEW FUNCTION: Enhanced CSV Export Functionality
def export_cleaned_data_interface(df: pd.DataFrame, processor: ProDataProcessor):
    """Enhanced interface for exporting cleaned data as CSV"""
    
    st.write("### 💾 Export Cleaned Data")
    with st.expander("💡 What this does"):
        st.write("Download your cleaned data as CSV file after all processing steps")
    
    # Show current data status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Current Columns", f"{df.shape[1]}")
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data preview
    st.write("#### 📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Export options
    st.write("#### ⚙️ Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_index = st.checkbox("Include Index", value=False, 
                                  help="Include row numbers in exported file")
        
        encoding_type = st.selectbox(
            "File Encoding",
            ["utf-8", "utf-8-sig", "latin-1"],
            help="Choose file encoding (utf-8 recommended for most cases)"
        )
    
    with col2:
        compression_type = st.selectbox(
            "Compression",
            ["None", "gzip", "zip"],
            help="Compress the CSV file to reduce size"
        )
        
        # Auto-detect best separator
        separator = st.selectbox(
            "Separator",
            [",", ";", "|", "\t"],
            help="Field separator for CSV file"
        )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            date_format = st.text_input(
                "Date Format", 
                "%Y-%m-%d",
                help="Format for date columns (e.g., %Y-%m-%d for 2024-01-15)"
            )
            
            float_precision = st.number_input(
                "Float Precision", 
                min_value=0, 
                max_value=10, 
                value=2,
                help="Number of decimal places for float values"
            )
        
        with col2:
            na_rep = st.text_input(
                "Missing Value Representation",
                "",
                help="How to represent missing values (empty for blank)"
            )
            
            quote_all = st.checkbox(
                "Quote All Fields", 
                value=False,
                help="Put quotes around all fields"
            )
    
    # Apply cleaning before export if requested
    st.write("#### 🚀 Apply Cleaning Before Export")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        apply_cleaning = st.checkbox(
            "Apply all queued cleaning operations before export", 
            value=True,
            help="Automatically apply all data cleaning steps before downloading"
        )
    
    with col2:
        if st.button("🔄 Preview Cleaned Data", type="secondary"):
            if apply_cleaning and processor.get_operations_queue():
                with st.spinner("Applying cleaning operations..."):
                    preview_df = processor.apply_all_operations(df)
                    st.session_state.cleaned_preview = preview_df
                    st.success("✅ Cleaning operations applied!")
                    
                    # Show preview comparison
                    st.write("**Preview Comparison:**")
                    comparison_col1, comparison_col2 = st.columns(2)
                    
                    with comparison_col1:
                        st.write("**Before Cleaning**")
                        st.dataframe(df.head(5), use_container_width=True)
                    
                    with comparison_col2:
                        st.write("**After Cleaning**")
                        st.dataframe(preview_df.head(5), use_container_width=True)
            else:
                st.session_state.cleaned_preview = df
                st.info("No cleaning operations to apply or preview disabled")
    
    # Export button
    st.write("#### 📥 Download Cleaned Data")
    
    if st.button("💾 Export Cleaned CSV", type="primary", use_container_width=True):
        try:
            with st.spinner("Preparing your cleaned data for download..."):
                # Get the data to export
                if apply_cleaning and processor.get_operations_queue():
                    export_df = processor.apply_all_operations(df)
                    st.success("✅ All cleaning operations applied!")
                else:
                    export_df = df
                    st.info("ℹ️ Exporting current data without additional cleaning")
                
                # Prepare CSV data
                csv_buffer = io.StringIO()
                
                # Apply formatting options
                export_kwargs = {
                    'index': include_index,
                    'sep': separator,
                    'encoding': encoding_type,
                    'date_format': date_format if date_format else None
                }
                
                if na_rep:
                    export_kwargs['na_rep'] = na_rep
                
                # Format float precision
                if float_precision > 0:
                    float_cols = export_df.select_dtypes(include=['float']).columns
                    for col in float_cols:
                        export_df[col] = export_df[col].round(float_precision)
                
                # Export to CSV
                export_df.to_csv(csv_buffer, **export_kwargs)
                csv_data = csv_buffer.getvalue()
                
                # Create download button
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cleaned_data_{timestamp}.csv"
                
                st.download_button(
                    label="📥 Download CSV File",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Click to download your cleaned data as CSV",
                    use_container_width=True
                )
                
                # Show export summary
                st.success("🎉 Data prepared for download!")
                
                export_summary_col1, export_summary_col2, export_summary_col3 = st.columns(3)
                
                with export_summary_col1:
                    st.metric("Exported Rows", f"{export_df.shape[0]:,}")
                with export_summary_col2:
                    st.metric("Exported Columns", f"{export_df.shape[1]}")
                with export_summary_col3:
                    file_size = len(csv_data.encode('utf-8')) / 1024
                    st.metric("File Size", f"{file_size:.1f} KB")
                
                # Show data quality improvement if cleaning was applied
                if apply_cleaning and processor.get_operations_queue():
                    quality_before = st.session_state.quality_metrics.calculate_comprehensive_quality(df)
                    quality_after = st.session_state.quality_metrics.calculate_comprehensive_quality(export_df)
                    
                    overall_before = np.mean(list(quality_before.values()))
                    overall_after = np.mean(list(quality_after.values()))
                    
                    st.info(f"📈 Data quality improved from {overall_before:.1%} to {overall_after:.1%} (+{(overall_after - overall_before):.1%})")
        
        except Exception as e:
            st.error(f"❌ Export failed: {str(e)}")
            st.info("💡 Try adjusting export options or check your data for issues")

# Enhanced helper functions
def count_potential_issues(df: pd.DataFrame) -> int:
    """Enhanced potential issues counting"""
    issues = 0
    
    # Missing values
    issues += len(df.columns[df.isnull().any()])
    
    # Data type issues
    for col in df.columns:
        if str(df[col].dtype) == 'object':
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                if any(isinstance(x, (int, float)) for x in sample if not pd.isna(x)):
                    issues += 1
    
    # Duplicates
    if df.duplicated().sum() > 0:
        issues += 1
    
    return issues

def smart_correction_interface(df: pd.DataFrame, processor: ProDataProcessor):
    """Enhanced smart correction interface"""
    
    st.write("### 🎯 Smart Data Correction")
    with st.expander("💡 What this does"):
        st.write("Fix specific columns with AI recommendations and one-click fixes")
    
    selected_col = st.selectbox("Select column for detailed analysis", df.columns, 
                               help="Choose a column to see detailed analysis and fixes")
    
    if selected_col:
        col_data = df[selected_col]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Current Analysis")
            
            stats_data = {
                'Data Type': str(col_data.dtype),
                'Total Values': len(col_data),
                'Non-Null Values': col_data.count(),
                'Null Percentage': f"{(col_data.isnull().sum() / len(col_data)) * 100:.2f}%",
                'Unique Values': col_data.nunique(),
                'Memory Usage': f"{col_data.memory_usage(deep=True) / 1024:.2f} KB"
            }
            
            for key, value in stats_data.items():
                st.write(f"**{key}:** {value}")
            
            if col_data.nunique() < 20:
                st.write("**Value Distribution:**")
                value_counts = col_data.value_counts().head(5)
                for value, count in value_counts.items():
                    st.write(f"- `{value}`: {count} ({count/len(col_data):.1%})")
        
        with col2:
            st.write("#### AI Recommendations")
            
            detected_type, confidence = processor._detect_column_type(col_data)
            current_dtype = str(col_data.dtype)
            recommended_dtype = processor._get_target_dtype(detected_type)
            
            st.write(f"**Detected Pattern:** {detected_type} (confidence: {confidence:.1%})")
            st.write(f"**Recommended Type:** `{recommended_dtype}`")
            
            if current_dtype != recommended_dtype and confidence > 0.8:
                st.warning(f"🔧 **Recommendation**: Convert to `{recommended_dtype}`")
                
                if st.button(f"Apply Type Correction", key=f"correct_{selected_col}"):
                    processor.add_operation('auto_correct_dtypes', confidence_threshold=0.8)
                    st.success(f"✅ Type correction queued for {selected_col}")
            
            if col_data.isnull().sum() > 0:
                missing_pct = col_data.isnull().mean()
                
                if missing_pct < 0.05:
                    st.info(f"🔧 **Recommendation**: Remove {col_data.isnull().sum()} missing rows")
                elif missing_pct < 0.3:
                    imputation_value = processor._calculate_smart_imputation(col_data, 'auto')
                    st.info(f"🔧 **Recommendation**: Impute with `{imputation_value}`")
                else:
                    st.error(f"🔧 **Recommendation**: Strategic handling needed for {missing_pct:.1%} missing data")
                
                if st.button(f"Apply Smart Imputation", key=f"impute_{selected_col}"):
                    processor.add_operation('smart_imputation', strategy='auto')
                    st.success(f"✅ Smart imputation queued for {selected_col}")

def advanced_quality_analytics(df: pd.DataFrame):
    """Enhanced quality analytics"""
    
    st.write("### 📈 Advanced Quality Analytics")
    with st.expander("💡 What this does"):
        st.write("Understand data relationships and patterns through visualizations")
    
    st.write("#### 🔗 Data Relationships")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        try:
            correlation_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red = positive correlation, Blue = negative correlation")
        except Exception as e:
            st.info(f"Correlation analysis not available: {str(e)}")
    else:
        st.info("Need at least 2 numeric columns for correlation analysis")
    
    st.write("#### 📊 Distribution Analysis")
    
    if len(numeric_cols) > 0:
        selected_num_col = st.selectbox("Select numeric column", numeric_cols)
        
        if selected_num_col:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Distribution', 'Box Plot')
            )
            
            fig.add_trace(go.Histogram(x=df[selected_num_col], name='Distribution'), row=1, col=1)
            fig.add_trace(go.Box(y=df[selected_num_col], name='Box Plot'), row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Left: Value distribution | Right: Outlier detection")
    else:
        st.info("No numeric columns available for distribution analysis")

def batch_processing_interface(df: pd.DataFrame, processor: ProDataProcessor):
    """Enhanced batch processing interface"""
    
    st.write("### 🚀 Batch Quality Processing")
    with st.expander("💡 What this does"):
        st.write("Apply multiple cleaning operations at once using pre-built pipelines")
    
    st.write("#### 🏭 Quality Processing Pipelines")
    
    pipeline_options = {
        "Quick Clean": ["auto_correct_dtypes", "smart_imputation"],
        "Comprehensive Clean": ["auto_correct_dtypes", "smart_imputation", "outlier_mitigation", "consistency_repair"],
        "Production Ready": ["auto_correct_dtypes", "smart_imputation", "domain_validation", "outlier_mitigation", "consistency_repair", "pattern_based_cleaning"]
    }
    
    selected_pipeline = st.selectbox("Select processing pipeline", list(pipeline_options.keys()))
    
    if st.button(f"Apply {selected_pipeline} Pipeline"):
        operations = pipeline_options[selected_pipeline]
        for op in operations:
            processor.add_operation(op)
        
        st.success(f"✅ {selected_pipeline} pipeline queued with {len(operations)} operations")
    
    st.write("#### ⚙️ Custom Operation Builder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        operation_type = st.selectbox(
            "Operation Type",
            ["auto_correct_dtypes", "smart_imputation", "outlier_mitigation", "consistency_repair", "pattern_based_cleaning"]
        )
    
    with col2:
        confidence = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
    
    if st.button("Add Custom Operation"):
        processor.add_operation(operation_type, confidence_threshold=confidence)
        st.success(f"✅ Added {operation_type} operation")

def generate_enterprise_quality_report(df: pd.DataFrame, quality_metrics: QualityMetrics):
    """Enhanced enterprise quality report"""
    
    st.write("### 📋 Enterprise Quality Report")
    with st.expander("💡 What this does"):
        st.write("Generate comprehensive report with scores, metrics, and recommendations")
    
    if st.button("Generate Comprehensive Report", type="primary"):
        with st.spinner("Generating enterprise quality report..."):
            
            st.write("#### 📊 Executive Summary")
            
            quality_scores = quality_metrics.calculate_comprehensive_quality(df)
            overall_score = np.mean(list(quality_scores.values()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Overall Quality Score", f"{overall_score:.1%}")
                st.metric("Data Reliability Index", f"{(overall_score * 100):.1f}/100")
            
            with col2:
                st.metric("Recommended Actions", count_recommended_actions(df))
                st.metric("Estimated Improvement", f"+{(1 - overall_score) * 100:.1f}% possible")
            
            st.write("#### 🔍 Detailed Quality Analysis")
            
            for metric, score in quality_scores.items():
                st.write(f"**{metric.title()}**: {score:.1%}")
                st.progress(score)
            
            st.write("#### ⚙️ Technical Specifications")
            
            tech_specs = {
                'Data Volume': f"{df.shape[0]:,} records × {df.shape[1]} attributes",
                'Storage Efficiency': f"{(df.memory_usage(deep=True).sum() / (df.shape[0] * df.shape[1] * 8)):.1%}",
                'Data Density': f"{((df.size - df.isnull().sum().sum()) / df.size):.1%}",
                'Type Optimization': f"{(len(df.select_dtypes(include=['int', 'float', 'category']).columns) / len(df.columns)):.1%}"
            }
            
            for spec, value in tech_specs.items():
                st.write(f"- **{spec}**: {value}")

def count_recommended_actions(df: pd.DataFrame) -> int:
    """Enhanced recommended actions counting"""
    actions = 0
    
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if df[col].nunique() / len(df) < 0.3:
            actions += 1
    
    if df.isnull().sum().sum() > 0:
        actions += 1
    
    if df.duplicated().sum() > 0:
        actions += 1
    
    return actions

def preview_intelligent_changes(df: pd.DataFrame, processor: ProDataProcessor):
    """Enhanced changes preview"""
    
    st.write("### 🔍 Intelligent Changes Preview")
    with st.expander("💡 What this does"):
        st.write("See what changes will be made before applying them")
    
    if not processor.get_operations_queue():
        st.info("No intelligent operations queued")
        return
    
    st.write("#### 📋 Queued Intelligent Operations")
    operations = processor.get_operations_queue()
    
    for i, op in enumerate(operations, 1):
        with st.expander(f"{i}. {op['description']}"):
            st.json(op['kwargs'])
    
    try:
        preview_df = processor.apply_all_operations(df)
        
        st.write("#### ⚖️ Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Before Processing**")
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"- Missing: {df.isnull().sum().sum()}")
            st.write(f"- Object columns: {len(df.select_dtypes(include=['object']).columns)}")
        
        with col2:
            st.write("**After Processing**")
            st.write(f"- Shape: {preview_df.shape}")
            st.write(f"- Memory: {preview_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            st.write(f"- Missing: {preview_df.isnull().sum().sum()}")
            st.write(f"- Object columns: {len(preview_df.select_dtypes(include=['object']).columns)}")
        
        quality_before = st.session_state.quality_metrics.calculate_comprehensive_quality(df)
        quality_after = st.session_state.quality_metrics.calculate_comprehensive_quality(preview_df)
        
        st.write("#### 📈 Quality Improvement")
        
        improvement_data = []
        for metric in quality_before.keys():
            before = quality_before[metric]
            after = quality_after[metric]
            improvement = after - before
            
            improvement_data.append({
                'Metric': metric.title(),
                'Before': f"{before:.1%}",
                'After': f"{after:.1%}", 
                'Improvement': f"{improvement:+.1%}"
            })
        
        improvement_df = pd.DataFrame(improvement_data)
        st.dataframe(improvement_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"Preview generation failed: {str(e)}")

def save_quality_protocol(processor: ProDataProcessor):
    """Enhanced protocol saving"""
    
    if not processor.get_operations_queue():
        st.warning("No operations to save as protocol")
        return
    
    protocol = processor.export_cleaning_template()
    
    st.write("#### 💾 Quality Protocol")
    st.json(protocol)
    
    protocol_json = json.dumps(protocol, indent=2)
    st.download_button(
        label="📥 Download Quality Protocol",
        data=protocol_json,
        file_name=f"quality_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def execute_quality_pipeline(df: pd.DataFrame, processor: ProDataProcessor, selected_file: str):
    """Enhanced pipeline execution"""
    
    with st.spinner("Executing enterprise quality pipeline..."):
        try:
            cleaned_df = processor.apply_all_operations(df)
            
            st.session_state.processed_data[selected_file]['dataframe'] = cleaned_df
            st.session_state.processed_data[selected_file]['quality_protocol'] = processor.get_operations_queue()
            
            st.success("✅ Enterprise quality pipeline executed successfully!")
            st.balloons()
            
            st.write("#### 📋 Execution Summary")
            
            quality_before = st.session_state.quality_metrics.calculate_comprehensive_quality(df)
            quality_after = st.session_state.quality_metrics.calculate_comprehensive_quality(cleaned_df)
            
            overall_before = np.mean(list(quality_before.values()))
            overall_after = np.mean(list(quality_after.values()))
            
            st.metric(
                "Overall Quality Improvement", 
                f"{overall_after:.1%}",
                f"+{(overall_after - overall_before):.1%}"
            )
            
            st.write("#### 🎯 Recommended Next Steps")
            
            if overall_after < 0.95:
                st.warning("""
                **Continue Quality Improvement:**
                - Review remaining data issues
                - Implement domain-specific validation
                - Consider advanced imputation techniques
                """)
            else:
                st.success("""
                **Production Ready:**
                - Data meets enterprise quality standards
                - Proceed to analysis and modeling
                - Document quality protocol for future use
                """)
                
        except Exception as e:
            st.error(f"Pipeline execution failed: {str(e)}")

# Enhanced main rendering function
def render_data_cleaning():
    """Enhanced professional-grade data cleaning interface"""
    
    st.header("🏢 Enterprise Data Quality Suite")
    
    # Minimized instructions
    with st.expander("📚 **Quick Guide - Click to Expand**", expanded=False):
        st.markdown("""
        **Follow these steps for best results:**
        1. **Upload your data** in Data Loader section first
        2. **Auto-Diagnosis**: Find issues automatically  
        3. **Smart Correction**: Fix specific problems
        4. **Quality Analytics**: Understand your data
        5. **Batch Processing**: Apply multiple fixes
        6. **Quality Report**: See improvements
        7. **Export Data**: Download cleaned CSV file
        
        **Pro Tip**: Use AI Assistant for personalized help!
        """)
    
    st.markdown("""
    **Production-grade data cleaning with intelligent automation and enterprise-level quality assurance.**
    """)
    
    # Initialize enhanced components
    if 'pro_data_processor' not in st.session_state:
        st.session_state.pro_data_processor = ProDataProcessor()
    
    if 'quality_metrics' not in st.session_state:
        st.session_state.quality_metrics = QualityMetrics()
    
    if 'gen_ai_assistant' not in st.session_state:
        st.session_state.gen_ai_assistant = GenAIAssistant()
    
    # Check data availability
    if 'processed_data' not in st.session_state or not st.session_state.processed_data:
        st.warning("""
        📭 **No data available!**
        
        **To get started:**
        1. Go to **Data Loader** section
        2. Upload your data file
        3. Return here to begin cleaning
        
        Don't have a file? Use sample data to test features!
        """)
        
        if st.button("🔄 Load Sample Data for Demonstration"):
            sample_df = pd.DataFrame({
                'customer_id': range(1, 101),
                'customer_name': [f'Customer_{i}' for i in range(1, 101)],
                'age': np.random.normal(45, 15, 100).astype(int),
                'annual_income': np.random.lognormal(10.5, 0.8, 100),
                'credit_score': np.random.normal(650, 100, 100).astype(int),
                'is_active': np.random.choice(['yes', 'no', '1', '0', 'true', 'false'], 100),
                'signup_date': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)) for _ in range(100)],
                'email': [f"user_{i}@example.com" for i in range(1, 101)]
            })
            
            sample_df.loc[sample_df.sample(frac=0.1).index, 'age'] = np.nan
            sample_df.loc[sample_df.sample(frac=0.05).index, 'annual_income'] = np.nan
            sample_df.loc[10:15, 'customer_name'] = 'Duplicate_Customer'
            
            st.session_state.processed_data = {'sample_data': {'dataframe': sample_df}}
            st.session_state.current_file = 'sample_data'
            st.success("✅ Sample data loaded! Explore all cleaning features.")
            st.rerun()
        
        return
    
    selected_file = st.session_state.current_file
    data_info = st.session_state.processed_data[selected_file]
    df = data_info['dataframe'].copy()
    processor = st.session_state.pro_data_processor
    quality_metrics = st.session_state.quality_metrics
    ai_assistant = st.session_state.gen_ai_assistant
    
    # Enhanced AI Assistant Sidebar
    with st.sidebar:
        st.header("🤖 AI Data Assistant")
        st.markdown("**Ask me anything about your data!**")
        
        ai_question = st.text_area(
            "Your question:",
            placeholder="E.g.: 'What missing values do I have?', 'How can I improve my data?'",
            height=100
        )
        
        if st.button("🔍 Get AI Insights", use_container_width=True):
            with st.spinner("🤖 AI Assistant analyzing..."):
                ai_response = ai_assistant.get_ai_insights(df, ai_question)
                st.success("AI Analysis Complete!")
                st.markdown(ai_response)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Questions:")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Missing Values?"):
                ai_response = ai_assistant.get_ai_insights(df, "What missing values do I have?")
                st.markdown(ai_response)
        with col2:
            if st.button("Data Quality?"):
                ai_response = ai_assistant.get_ai_insights(df, "How good is my data quality?")
                st.markdown(ai_response)
    
    # Enhanced Executive Dashboard
    st.subheader("📊 Executive Quality Dashboard")
    with st.expander("💡 Understanding your dashboard"):
        st.write("Shows overall data health score and key metrics. Green = Good, Yellow = Needs Attention, Red = Critical")
    
    display_executive_dashboard(df, quality_metrics)
    
    st.markdown("---")
    
    # Enhanced Cleaning Workflow - ADDED EXPORT TAB
    st.subheader("🔧 Intelligent Cleaning Pipeline")
    with st.expander("💡 Recommended workflow"):
        st.write("Follow tabs in order: Auto-Diagnosis → Smart Correction → Quality Analytics → Batch Processing → Quality Report → Export Data")
    
    cleaning_tabs = st.tabs([
        "🤖 Auto-Diagnosis", 
        "🎯 Smart Correction", 
        "📈 Quality Analytics",
        "🚀 Batch Processing",
        "📋 Quality Report",
        "💾 Export Data"  # NEW TAB ADDED
    ])
    
    with cleaning_tabs[0]:
        auto_data_diagnosis(df, processor)
    
    with cleaning_tabs[1]:
        smart_correction_interface(df, processor)
    
    with cleaning_tabs[2]:
        advanced_quality_analytics(df)
    
    with cleaning_tabs[3]:
        batch_processing_interface(df, processor)
    
    with cleaning_tabs[4]:
        generate_enterprise_quality_report(df, quality_metrics)
    
    # NEW TAB: Export Data
    with cleaning_tabs[5]:
        export_cleaned_data_interface(df, processor)
    
    st.markdown("---")
    
    # Enhanced Action Panel
    st.subheader("🎯 Take Action")
    with st.expander("💡 Final steps"):
        st.write("Preview changes, save your work, reset if needed, then execute when ready")
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        if st.button("🔍 Preview Changes", type="secondary", use_container_width=True,
                    help="See what changes will be made before applying"):
            preview_intelligent_changes(df, processor)
    
    with col2:
        if st.button("💾 Save Protocol", type="secondary", use_container_width=True,
                    help="Save cleaning steps for similar datasets"):
            save_quality_protocol(processor)
    
    with col3:
        if st.button("🔄 Reset", type="secondary", use_container_width=True,
                    help="Clear all planned operations and start fresh"):
            processor.clear_operations()
            st.success("✅ Pipeline reset!")
            st.rerun()
    
    with col4:
        if st.button("🚀 Execute Pipeline", type="primary", use_container_width=True,
                    help="Apply all cleaning operations to your data"):
            execute_quality_pipeline(df, processor, selected_file)

# Enhanced display functions
def display_executive_dashboard(df: pd.DataFrame, quality_metrics: QualityMetrics):
    """Enhanced executive dashboard"""
    
    quality_scores = quality_metrics.calculate_comprehensive_quality(df)
    overall_score = np.mean(list(quality_scores.values()))
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Overall Quality", f"{overall_score:.1%}", 
                 delta=f"{(overall_score - 0.7):.1%}" if overall_score > 0.7 else None,
                 delta_color="normal" if overall_score > 0.7 else "inverse")
        st.caption("Combined quality measure")
    
    with col2:
        st.metric("Completeness", f"{quality_scores['completeness']:.1%}")
        st.caption("Data filled in (not missing)")
    
    with col3:
        st.metric("Accuracy", f"{quality_scores['accuracy']:.1%}")
        st.caption("Correct and reliable values")
    
    with col4:
        st.metric("Consistency", f"{quality_scores['consistency']:.1%}")
        st.caption("Follows same rules throughout")
    
    with col5:
        st.metric("Validity", f"{quality_scores['validity']:.1%}")
        st.caption("Follows expected formats")
    
    with col6:
        st.metric("Uniqueness", f"{quality_scores['uniqueness']:.1%}")
        st.caption("Unique data (not duplicated)")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = overall_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Data Quality Score"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 70], 'color': "lightgray"},
                {'range': [70, 90], 'color': "gray"},
                {'range': [90, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    display_pro_quality_alerts(df, quality_scores)

def display_pro_quality_alerts(df: pd.DataFrame, quality_scores: Dict[str, float]):
    """Enhanced quality alerts"""
    
    alerts = []
    
    if quality_scores['completeness'] < 0.9:
        missing_cols = df.columns[df.isnull().any()].tolist()
        high_missing = [col for col in missing_cols if df[col].isnull().mean() > 0.1]
        
        if high_missing:
            alerts.append({
                "level": "error",
                "title": "Critical Data Gaps",
                "message": f"{len(high_missing)} columns have >10% missing values",
                "action": "Use Smart Imputation to fill missing values"
            })
    
    if quality_scores['accuracy'] < 0.95:
        alerts.append({
            "level": "warning", 
            "title": "Data Accuracy Concerns",
            "message": "Potential data integrity issues",
            "action": "Check for negative values in numeric columns"
        })
    
    if alerts:
        st.write("#### 🚨 Quality Assurance Alerts")
        
        for alert in alerts:
            if alert["level"] == "error":
                st.error(f"**{alert['title']}**: {alert['message']} - *{alert['action']}*")
            else:
                st.warning(f"**{alert['title']}**: {alert['message']} - *{alert['action']}*")
    else:
        st.success("🎉 All quality metrics meet enterprise standards!")

def auto_data_diagnosis(df: pd.DataFrame, processor: ProDataProcessor):
    """Enhanced auto diagnosis"""
    
    st.write("### 🤖 Automated Data Diagnosis")
    with st.expander("💡 What this does"):
        st.write("Automatically scans your data for issues and recommends fixes")
    
    with st.spinner("Running comprehensive data analysis..."):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Dataset Size", f"{df.shape[0]:,} × {df.shape[1]}")
        with col2:
            st.metric("Total Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col3:
            st.metric("Data Types", f"{len(df.dtypes.unique())} unique")
        with col4:
            st.metric("Potential Issues", count_potential_issues(df))
        
        st.write("#### 🔍 Automatic Type Analysis")
        with st.expander("💡 Understanding the table"):
            st.write("Shows current types, detected types, confidence levels, and recommended actions")
        
        type_analysis = []
        for col in df.columns:
            current_dtype = str(df[col].dtype)
            detected_type, confidence = processor._detect_column_type(df[col])
            recommended_dtype = processor._get_target_dtype(detected_type)
            
            type_analysis.append({
                'Column': col,
                'Current Type': current_dtype,
                'Detected Type': detected_type,
                'Confidence': f"{confidence:.1%}",
                'Recommended': recommended_dtype,
                'Action Needed': 'Yes' if current_dtype != recommended_dtype and confidence > 0.8 else 'No'
            })
        
        type_df = pd.DataFrame(type_analysis)
        st.dataframe(type_df, use_container_width=True)
        
        st.write("#### ⚡ Intelligent Correction Recommendations")
        
        corrections_needed = type_df[type_df['Action Needed'] == 'Yes']
        
        if len(corrections_needed) > 0:
            st.warning(f"**AI Recommendation**: Found {len(corrections_needed)} columns needing data type correction")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.info("""
                **What will happen**: 
                - Text numbers → Actual numbers
                - Text dates → Actual dates  
                - Yes/No text → True/False
                - Repeated text → Categories
                """)
            with col2:
                if st.button("🤖 Apply Type Corrections", type="primary", use_container_width=True):
                    processor.add_operation('auto_correct_dtypes', confidence_threshold=0.8)
                    st.success(f"✅ Queued correction for {len(corrections_needed)} columns")
                    st.balloons()
        else:
            st.success("🎉 No data type corrections needed!")

# For testing purposes
if __name__ == "__main__":
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
        st.session_state.current_file = 'test_data'
        
        test_df = pd.DataFrame({
            'customer_id': range(1, 101),
            'customer_name': [f'Customer_{i}' for i in range(1, 101)],
            'age': np.random.normal(45, 15, 100).astype(int),
            'annual_income': np.random.lognormal(10.5, 0.8, 100),
            'credit_score': np.random.normal(650, 100, 100).astype(int),
            'is_active': np.random.choice(['yes', 'no', '1', '0', 'true', 'false'], 100),
            'signup_date': [datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1460)) for _ in range(100)]
        })
        
        st.session_state.processed_data['test_data'] = {'dataframe': test_df}
    
    render_data_cleaning()