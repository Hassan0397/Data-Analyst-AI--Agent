import pandas as pd
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression
import openai
import warnings
import logging
from datetime import datetime
import re

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnterpriseEDAAnalyzer:
    """Enterprise-grade Exploratory Data Analysis with enhanced AI capabilities"""
    
    def __init__(self, df: pd.DataFrame, api_key: str = None):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Enhanced initialization
        self.analysis_history = []
        self.insight_cache = {}
        
        # Initialize LLM client if API key provided with enhanced configuration
        self.llm_client = None
        if api_key:
            try:
                # Validate API key format (basic check)
                if self._validate_api_key(api_key):
                    self.llm_client = openai.OpenAI(api_key=api_key)
                    logger.info("OpenAI client initialized successfully")
                else:
                    logger.warning("Invalid API key format provided")
                    print("Warning: Invalid API key format. AI features will be disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                print(f"Warning: Failed to initialize OpenAI client: {e}")
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Basic validation of OpenAI API key format"""
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Basic format check - OpenAI keys typically start with 'sk-'
        if api_key.startswith('sk-'):
            return True
        
        # Also allow other formats that might be valid
        if len(api_key) > 20:
            return True
            
        return False
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistical summary with enhanced metrics"""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_percentage = (self.df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
        
        # Enhanced metrics
        numeric_stats = {}
        if self.numeric_cols:
            numeric_stats = {
                'numeric_mean_range': [self.df[col].mean() for col in self.numeric_cols[:3]],
                'has_negative_values': any(self.df[col].min() < 0 for col in self.numeric_cols),
                'zero_percentage': (self.df[self.numeric_cols] == 0).sum().sum() / total_cells * 100
            }
        
        return {
            'dataset_shape': self.df.shape,
            'numeric_columns': self.numeric_cols,
            'categorical_columns': self.categorical_cols,
            'date_columns': self.date_cols,
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percentage': missing_percentage,
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'total_columns': len(self.df.columns),
            'total_rows': len(self.df),
            'column_types': dict(self.df.dtypes.value_counts()),
            **numeric_stats
        }
    
    def get_enhanced_column_info(self) -> pd.DataFrame:
        """Get enhanced column information with data quality metrics"""
        col_info = []
        
        for col in self.df.columns:
            col_data = self.df[col]
            null_count = col_data.isnull().sum()
            unique_count = col_data.nunique()
            total_count = len(col_data)
            
            info = {
                'Column': col,
                'Type': str(col_data.dtype),
                'Non-Null': total_count - null_count,
                'Null': null_count,
                'Null %': (null_count / total_count) * 100,
                'Unique': unique_count,
                'Unique %': (unique_count / total_count) * 100 if total_count > 0 else 0
            }
            
            # Numeric specific metrics
            if col in self.numeric_cols:
                info.update({
                    'Mean': col_data.mean(),
                    'Std': col_data.std(),
                    'Min': col_data.min(),
                    'Max': col_data.max()
                })
            else:
                info.update({
                    'Most Frequent': col_data.mode().iloc[0] if not col_data.empty else 'N/A',
                    'Top Frequency': col_data.value_counts().iloc[0] if not col_data.empty else 0
                })
            
            col_info.append(info)
        
        return pd.DataFrame(col_info)
    
    def calculate_data_quality_score(self) -> Dict[str, float]:
        """Calculate comprehensive data quality score"""
        basic_stats = self.get_basic_stats()
        
        # Completeness score (0-10)
        completeness = max(0, 10 - (basic_stats['missing_percentage'] / 2))
        
        # Accuracy score (based on data patterns)
        accuracy_indicators = 0
        total_indicators = 0
        
        # Check for potential data quality issues
        for col in self.numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                total_indicators += 1
                # Check for outliers (moderate amount is good, too many is bad)
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                outlier_percentage = outliers / len(col_data) * 100
                
                if outlier_percentage < 5:
                    accuracy_indicators += 1
                elif outlier_percentage > 20:
                    accuracy_indicators += 0
                else:
                    accuracy_indicators += 0.5
        
        accuracy = (accuracy_indicators / total_indicators * 10) if total_indicators > 0 else 8
        
        # Consistency score
        consistency = 8
        
        # Timeliness score (placeholder)
        timeliness = 9
        
        # Validity score
        validity_indicators = 0
        total_indicators = 0
        
        for col in self.numeric_cols:
            total_indicators += 1
            if self.df[col].notna().any():
                if (self.df[col] >= 0).all() and col.lower() in ['age', 'price', 'quantity']:
                    validity_indicators += 1
        
        validity = (validity_indicators / total_indicators * 10) if total_indicators > 0 else 7
        
        # Overall score (weighted average)
        weights = {'completeness': 0.3, 'accuracy': 0.25, 'consistency': 0.2, 'timeliness': 0.15, 'validity': 0.1}
        overall_score = (
            completeness * weights['completeness'] +
            accuracy * weights['accuracy'] +
            consistency * weights['consistency'] +
            timeliness * weights['timeliness'] +
            validity * weights['validity']
        )
        
        return {
            'completeness': round(completeness, 1),
            'accuracy': round(accuracy, 1),
            'consistency': round(consistency, 1),
            'timeliness': round(timeliness, 1),
            'validity': round(validity, 1),
            'overall_score': round(overall_score, 1)
        }
    
    def generate_data_quality_recommendations(self) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []
        basic_stats = self.get_basic_stats()
        quality_score = self.calculate_data_quality_score()
        
        if basic_stats['missing_values'] > 0:
            recommendations.append(
                f"Address {basic_stats['missing_values']} missing values through imputation or data collection improvements"
            )
        
        if basic_stats['duplicate_rows'] > 0:
            recommendations.append(
                f"Remove or investigate {basic_stats['duplicate_rows']} duplicate rows"
            )
        
        if quality_score['completeness'] < 7:
            recommendations.append("Implement data validation rules to improve data completeness")
        
        if quality_score['accuracy'] < 7:
            recommendations.append("Establish data quality monitoring for outlier detection and correction")
        
        if len(self.numeric_cols) == 0:
            recommendations.append("Consider converting some categorical data to numeric for better analysis")
        
        return recommendations
    
    def analyze_distributions(self) -> Dict[str, Any]:
        """Enhanced distribution analysis with additional metrics"""
        distributions = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) == 0:
                distributions[col] = {'error': 'No valid data'}
                continue
                
            # Enhanced statistical measures
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            # Advanced metrics
            cv = (data.std() / data.mean()) * 100 if data.mean() != 0 else float('inf')
            mad = np.mean(np.abs(data - data.mean()))
            
            distributions[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'variance': float(data.var()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'range': [float(data.min()), float(data.max())],
                'iqr': float(IQR),
                'outliers': int(self._count_outliers(data)),
                'normality_test': self._test_normality(data),
                'percentiles': {
                    '1st': float(data.quantile(0.01)),
                    '5th': float(data.quantile(0.05)),
                    '25th': float(Q1),
                    '75th': float(Q3),
                    '95th': float(data.quantile(0.95)),
                    '99th': float(data.quantile(0.99))
                },
                'cv': float(cv),
                'mad': float(mad),
                'is_normal': self._is_approximately_normal(data),
                'data_type': 'continuous' if data.nunique() > 20 else 'discrete'
            }
        
        return distributions
    
    def _is_approximately_normal(self, data: pd.Series, alpha: float = 0.05) -> bool:
        """Check if data is approximately normal using multiple tests"""
        if len(data) < 3:
            return False
        
        try:
            # Shapiro-Wilk test
            _, shapiro_p = stats.shapiro(data)
            # D'Agostino test
            _, dagostino_p = stats.normaltest(data)
            
            return shapiro_p > alpha and dagostino_p > alpha
        except:
            return False
    
    def _count_outliers(self, data: pd.Series) -> int:
        """Enhanced outlier detection with multiple methods"""
        if len(data) == 0:
            return 0
        
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
        
        # Z-score method for larger datasets
        zscore_outliers = 0
        if len(data) > 30:
            z_scores = np.abs(stats.zscore(data))
            zscore_outliers = (z_scores > 3).sum()
        
        return max(iqr_outliers, zscore_outliers)
    
    def _test_normality(self, data: pd.Series) -> Dict[str, Any]:
        """Enhanced normality testing with additional tests"""
        if len(data) < 3:
            return {'error': 'Insufficient data for normality tests'}
        
        try:
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = stats.shapiro(data)
            
            # D'Agostino's test
            dagostino_stat, dagostino_p = stats.normaltest(data)
            
            # Anderson-Darling test
            anderson_result = stats.anderson(data, dist='norm')
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
            
            return {
                'shapiro_wilk': {
                    'statistic': float(shapiro_stat),
                    'p_value': float(shapiro_p),
                    'is_normal': shapiro_p > 0.05
                },
                'dagostino': {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'is_normal': dagostino_p > 0.05
                },
                'anderson': {
                    'statistic': float(anderson_result.statistic),
                    'critical_values': anderson_result.critical_values.tolist(),
                    'significance_level': anderson_result.significance_level.tolist()
                },
                'kolmogorov_smirnov': {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'is_normal': ks_p > 0.05
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """Enhanced correlation analysis with multiple methods"""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
        
        try:
            if method == 'all':
                # Return correlations using multiple methods
                pearson_corr = self.df[self.numeric_cols].corr(method='pearson')
                spearman_corr = self.df[self.numeric_cols].corr(method='spearman')
                kendall_corr = self.df[self.numeric_cols].corr(method='kendall')
                
                return {
                    'pearson': pearson_corr,
                    'spearman': spearman_corr,
                    'kendall': kendall_corr
                }
            else:
                return self.df[self.numeric_cols].corr(method=method)
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return pd.DataFrame()
    
    def get_strong_correlations(self, threshold: float = 0.7) -> pd.DataFrame:
        """Enhanced strong correlation detection"""
        corr_matrix = self.analyze_correlations('pearson')
        strong_corrs = []
        
        if corr_matrix.empty:
            return pd.DataFrame()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    # Enhanced strength classification
                    if abs(corr_val) > 0.9:
                        strength = 'very strong'
                        implication = 'Potential multicollinearity'
                    elif abs(corr_val) > 0.7:
                        strength = 'strong'
                        implication = 'Strong relationship'
                    elif abs(corr_val) > 0.5:
                        strength = 'moderate'
                        implication = 'Moderate relationship'
                    else:
                        strength = 'weak'
                        implication = 'Weak relationship'
                    
                    strong_corrs.append({
                        'variable_1': corr_matrix.columns[i],
                        'variable_2': corr_matrix.columns[j],
                        'correlation': float(corr_val),
                        'abs_correlation': abs(corr_val),
                        'strength': strength,
                        'implication': implication,
                        'potential_causality': self._assess_causality_potential(corr_matrix.columns[i], corr_matrix.columns[j])
                    })
        
        result_df = pd.DataFrame(strong_corrs)
        if not result_df.empty:
            result_df = result_df.sort_values('abs_correlation', ascending=False)
        
        return result_df
    
    def _assess_causality_potential(self, var1: str, var2: str) -> str:
        """Simple assessment of potential causality based on variable names"""
        temporal_indicators = ['time', 'date', 'year', 'month', 'day', 'hour', 'minute']
        
        var1_lower = var1.lower()
        var2_lower = var2.lower()
        
        # Check if one variable appears to be temporal
        var1_temporal = any(indicator in var1_lower for indicator in temporal_indicators)
        var2_temporal = any(indicator in var2_lower for indicator in temporal_indicators)
        
        if var1_temporal and not var2_temporal:
            return f"{var2} might influence {var1} over time"
        elif var2_temporal and not var1_temporal:
            return f"{var1} might influence {var2} over time"
        else:
            return "Relationship direction unclear"
    
    def analyze_categorical(self) -> Dict[str, Any]:
        """Enhanced categorical analysis with business metrics"""
        categorical_analysis = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            value_percentages = (value_counts / len(self.df)) * 100
            
            # Enhanced concentration metrics
            top_3_percentage = value_percentages.head(3).sum()
            top_5_percentage = value_percentages.head(5).sum()
            gini = self._calculate_gini_coefficient(value_counts)
            entropy = self._calculate_entropy(value_counts)
            
            # Business relevance indicators
            business_relevance = self._assess_business_relevance(col, value_counts)
            
            categorical_analysis[col] = {
                'unique_count': int(self.df[col].nunique()),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'most_frequent_percentage': float(value_percentages.iloc[0]) if len(value_percentages) > 0 else 0,
                'value_distribution': value_counts.to_dict(),
                'entropy': float(entropy),
                'gini_coefficient': float(gini),
                'top_3_concentration': float(top_3_percentage),
                'top_5_concentration': float(top_5_percentage),
                'cardinality': 'high' if self.df[col].nunique() > 50 else 'medium' if self.df[col].nunique() > 10 else 'low',
                'business_relevance': business_relevance,
                'recommendation': self._generate_categorical_recommendation(col, value_counts)
            }
        
        return categorical_analysis
    
    def _assess_business_relevance(self, col: str, value_counts: pd.Series) -> str:
        """Assess business relevance of categorical column"""
        col_lower = col.lower()
        
        # Common business-relevant column patterns
        customer_indicators = ['customer', 'client', 'user', 'account']
        product_indicators = ['product', 'item', 'sku', 'category']
        location_indicators = ['region', 'country', 'state', 'city', 'location']
        time_indicators = ['month', 'year', 'quarter', 'season']
        
        if any(indicator in col_lower for indicator in customer_indicators):
            return 'Customer segmentation'
        elif any(indicator in col_lower for indicator in product_indicators):
            return 'Product analysis'
        elif any(indicator in col_lower for indicator in location_indicators):
            return 'Geographic analysis'
        elif any(indicator in col_lower for indicator in time_indicators):
            return 'Temporal analysis'
        else:
            return 'General analysis'
    
    def _generate_categorical_recommendation(self, col: str, value_counts: pd.Series) -> str:
        """Generate recommendations for categorical column handling"""
        unique_count = len(value_counts)
        total_count = value_counts.sum()
        
        if unique_count == 1:
            return "Consider removing - no variance"
        elif unique_count == 2:
            return "Good for binary classification"
        elif unique_count <= 10:
            return "Ideal for grouping and analysis"
        elif unique_count <= 50:
            return "Consider top-N grouping for analysis"
        else:
            return "High cardinality - consider encoding or grouping"
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of categorical distribution"""
        probabilities = value_counts / value_counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
    
    def _calculate_gini_coefficient(self, value_counts: pd.Series) -> float:
        """Calculate Gini coefficient for concentration measurement"""
        proportions = value_counts / value_counts.sum()
        return float(1 - np.sum(proportions ** 2))
    
    def detect_anomalies(self, method: str = 'iqr') -> pd.DataFrame:
        """Enhanced anomaly detection with multiple methods"""
        anomalies = pd.DataFrame()
        
        if method == 'iqr':
            for col in self.numeric_cols:
                col_anomalies = self._detect_iqr_anomalies(col)
                anomalies = pd.concat([anomalies, col_anomalies], ignore_index=True)
        
        elif method == 'isolation_forest':
            anomalies = self._detect_isolation_forest_anomalies()
        
        elif method == 'ensemble':
            # Use multiple methods and consensus
            iqr_anomalies = self.detect_anomalies('iqr')
            iso_anomalies = self._detect_isolation_forest_anomalies()
            
            # Combine results
            if not iqr_anomalies.empty and not iso_anomalies.empty:
                consensus_anomalies = pd.merge(iqr_anomalies, iso_anomalies, 
                                             on=list(self.df.columns), 
                                             how='inner')
                anomalies = consensus_anomalies
        
        return anomalies
    
    def _detect_iqr_anomalies(self, col: str) -> pd.DataFrame:
        """Enhanced IQR anomaly detection"""
        data = self.df[col].dropna()
        if len(data) == 0:
            return pd.DataFrame()
            
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        anomaly_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        anomalies = self.df[anomaly_mask].copy()
        
        if not anomalies.empty:
            anomalies['anomaly_column'] = col
            anomalies['anomaly_type'] = 'outlier'
            anomalies['anomaly_value'] = anomalies[col]
            anomalies['expected_range'] = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
            anomalies['deviation'] = anomalies[col].apply(
                lambda x: (x - upper_bound) if x > upper_bound else (lower_bound - x)
            )
            anomalies['severity'] = anomalies['deviation'].apply(
                lambda x: 'high' if x > 3 * IQR else 'medium' if x > 2 * IQR else 'low'
            )
        
        return anomalies
    
    def _detect_isolation_forest_anomalies(self) -> pd.DataFrame:
        """Enhanced Isolation Forest anomaly detection"""
        if len(self.numeric_cols) < 2:
            return pd.DataFrame()
            
        numeric_data = self.df[self.numeric_cols].fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Adaptive contamination based on data size
        contamination = min(0.1, 50 / len(self.df))
        
        iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=42,
            n_estimators=100
        )
        anomaly_labels = iso_forest.fit_predict(scaled_data)
        anomaly_scores = iso_forest.decision_function(scaled_data)
        
        anomalies = self.df[anomaly_labels == -1].copy()
        if not anomalies.empty:
            anomalies['anomaly_type'] = 'isolation_forest'
            anomalies['anomaly_score'] = anomaly_scores[anomaly_labels == -1]
            anomalies['severity'] = anomalies['anomaly_score'].apply(
                lambda x: 'high' if x < -0.5 else 'medium' if x < -0.2 else 'low'
            )
        
        return anomalies
    
    def perform_enhanced_advanced_analysis(self) -> Dict[str, Any]:
        """Perform enterprise-level multivariate analysis with enhanced methods"""
        results = {}
        
        # Enhanced PCA for dimensionality reduction
        if len(self.numeric_cols) >= 3:
            numeric_data = self.df[self.numeric_cols].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            n_components = min(10, len(self.numeric_cols))
            
            # Use incremental PCA for large datasets
            if len(self.df) > 10000:
                pca = IncrementalPCA(n_components=n_components)
            else:
                pca = PCA(n_components=n_components, random_state=42)
            
            pca_result = pca.fit_transform(scaled_data)
            
            results['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'feature_names': self.numeric_cols,
                'n_components_80pct': len([x for x in np.cumsum(pca.explained_variance_ratio_) if x < 0.8]) + 1,
                'n_components_90pct': len([x for x in np.cumsum(pca.explained_variance_ratio_) if x < 0.9]) + 1
            }
        
        # Enhanced Clustering analysis
        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # Determine optimal number of clusters
            n_clusters = self._determine_optimal_clusters(scaled_data)
            
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(scaled_data)
                
                # Calculate multiple clustering metrics
                silhouette_avg = silhouette_score(scaled_data, cluster_labels) if len(set(cluster_labels)) > 1 else 0
                calinski_harabasz = calinski_harabasz_score(scaled_data, cluster_labels) if len(set(cluster_labels)) > 1 else 0
                
                results['clustering'] = {
                    'inertia': float(kmeans.inertia_),
                    'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict(),
                    'silhouette_score': float(silhouette_avg),
                    'calinski_harabasz_score': float(calinski_harabasz),
                    'n_clusters': n_clusters,
                    'cluster_centers': kmeans.cluster_centers_.tolist()
                }
        
        # Enhanced anomaly detection
        if len(self.numeric_cols) >= 2:
            numeric_data = self.df[self.numeric_cols].fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = iso_forest.fit_predict(scaled_data)
            
            results['anomaly_detection'] = {
                'anomaly_percentage': float((anomaly_scores == -1).mean() * 100),
                'anomaly_indices': np.where(anomaly_scores == -1)[0].tolist(),
                'anomaly_scores': iso_forest.decision_function(scaled_data).tolist(),
                'high_confidence_anomalies': np.where(iso_forest.decision_function(scaled_data) < -0.3)[0].tolist()
            }
        
        # Feature importance analysis
        if len(self.numeric_cols) >= 2:
            results['feature_importance'] = self._calculate_comprehensive_feature_importance()
        
        return results
    
    def _determine_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Determine optimal number of clusters using elbow method and silhouette analysis"""
        if len(data) < 10:
            return min(2, len(data))
        
        max_k = min(max_k, len(data) // 10)
        if max_k < 2:
            return 2
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)
            
            if len(set(labels)) > 1:
                silhouette_scores.append(silhouette_score(data, labels))
            else:
                silhouette_scores.append(0)
        
        # Simple elbow detection
        optimal_k = 2
        if len(inertias) > 1:
            # Find the point where the decrease in inertia slows down
            reductions = [inertias[i-1] - inertias[i] for i in range(1, len(inertias))]
            if reductions:
                avg_reduction = np.mean(reductions)
                for i, reduction in enumerate(reductions):
                    if reduction < avg_reduction * 0.5:
                        optimal_k = i + 2
                        break
        
        return min(optimal_k, max_k)
    
    def _calculate_comprehensive_feature_importance(self) -> Dict[str, Any]:
        """Calculate feature importance using multiple methods"""
        if len(self.numeric_cols) < 2:
            return {}
        
        # Use the first numeric column as target for demonstration
        target_col = self.numeric_cols[0]
        feature_cols = [col for col in self.numeric_cols if col != target_col]
        
        if not feature_cols:
            return {}
        
        X = self.df[feature_cols].fillna(0)
        y = self.df[target_col].fillna(0)
        
        results = {}
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        results['random_forest'] = rf_importance.to_dict('records')
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        results['mutual_information'] = mi_importance.to_dict('records')
        
        # Correlation-based importance
        corr_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': [abs(self.df[col].corr(self.df[target_col])) for col in feature_cols]
        }).sort_values('importance', ascending=False)
        results['correlation'] = corr_importance.to_dict('records')
        
        return results
    
    def calculate_feature_importance(self, target_col: str, feature_cols: List[str]) -> Dict[str, pd.DataFrame]:
        """Calculate feature importance for specific target"""
        X = self.df[feature_cols].fillna(0)
        y = self.df[target_col].fillna(0)
        
        results = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        results['Random Forest'] = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(random_state=42)
        gb.fit(X, y)
        results['Gradient Boosting'] = pd.DataFrame({
            'feature': feature_cols,
            'importance': gb.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        results['Mutual Information'] = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=True)
        
        return results
    
    def interpret_pca_components(self, pca_results: Dict[str, Any]) -> List[str]:
        """Interpret PCA components in business terms"""
        insights = []
        
        n_components_80 = pca_results.get('n_components_80pct', 0)
        n_components_90 = pca_results.get('n_components_90pct', 0)
        
        insights.append(f"Data can be summarized with {n_components_80} components while retaining 80% of information")
        insights.append(f"{n_components_90} components needed to capture 90% of data variance")
        
        # Analyze first two components
        if 'components' in pca_results and len(pca_results['components']) >= 2:
            comp1 = pca_results['components'][0]
            comp2 = pca_results['components'][1]
            
            # Find top features for each component
            feature_importance_1 = sorted(zip(pca_results['feature_names'], comp1), 
                                        key=lambda x: abs(x[1]), reverse=True)[:3]
            feature_importance_2 = sorted(zip(pca_results['feature_names'], comp2), 
                                        key=lambda x: abs(x[1]), reverse=True)[:3]
            
            insights.append(f"First component dominated by: {', '.join([f[0] for f in feature_importance_1])}")
            insights.append(f"Second component influenced by: {', '.join([f[0] for f in feature_importance_2])}")
        
        return insights
    
    def interpret_clusters(self, cluster_info: Dict[str, Any]) -> List[str]:
        """Interpret clustering results"""
        insights = []
        
        n_clusters = cluster_info.get('n_clusters', 0)
        silhouette_score = cluster_info.get('silhouette_score', 0)
        
        insights.append(f"Identified {n_clusters} natural groupings in the data")
        
        if silhouette_score > 0.7:
            insights.append("Excellent cluster separation - clear distinct groups")
        elif silhouette_score > 0.5:
            insights.append("Reasonable cluster structure - groups are distinguishable")
        elif silhouette_score > 0.25:
            insights.append("Weak cluster structure - groups overlap somewhat")
        else:
            insights.append("Poor clustering - data may not have clear natural groups")
        
        # Cluster size analysis
        cluster_sizes = cluster_info.get('cluster_sizes', {})
        if cluster_sizes:
            largest_cluster = max(cluster_sizes.values())
            smallest_cluster = min(cluster_sizes.values())
            size_ratio = largest_cluster / smallest_cluster if smallest_cluster > 0 else float('inf')
            
            if size_ratio > 10:
                insights.append("Significant cluster size imbalance - one dominant group")
            elif size_ratio > 3:
                insights.append("Moderate cluster size variation")
            else:
                insights.append("Balanced cluster sizes")
        
        return insights
    
    def interpret_anomalies(self, anomalies: pd.DataFrame) -> List[str]:
        """Interpret anomaly detection results"""
        insights = []
        
        if anomalies.empty:
            insights.append("No significant anomalies detected - data appears normal")
            return insights
        
        n_anomalies = len(anomalies)
        total_rows = len(self.df)
        anomaly_percentage = (n_anomalies / total_rows) * 100
        
        insights.append(f"Found {n_anomalies} anomalies ({anomaly_percentage:.1f}% of data)")
        
        if anomaly_percentage > 5:
            insights.append("High anomaly rate - may indicate data quality issues or unusual business conditions")
        elif anomaly_percentage > 1:
            insights.append("Moderate anomaly rate - typical for real-world datasets")
        else:
            insights.append("Low anomaly rate - data appears clean")
        
        # Analyze anomaly severity
        if 'severity' in anomalies.columns:
            severity_counts = anomalies['severity'].value_counts()
            if 'high' in severity_counts:
                insights.append(f"{severity_counts['high']} high-severity anomalies need immediate attention")
        
        return insights
    
    def generate_statistical_insights(self) -> List[str]:
        """Generate automated statistical insights"""
        insights = []
        basic_stats = self.get_basic_stats()
        distributions = self.analyze_distributions()
        
        # Data size insights
        if basic_stats['total_rows'] > 1000000:
            insights.append("Large dataset - sufficient for robust statistical analysis")
        elif basic_stats['total_rows'] > 10000:
            insights.append("Moderate dataset size - good for most analytical purposes")
        else:
            insights.append("Small dataset - consider collecting more data for robust insights")
        
        # Data quality insights
        if basic_stats['missing_percentage'] > 20:
            insights.append("High missing data rate - consider data imputation strategies")
        elif basic_stats['missing_percentage'] > 5:
            insights.append("Moderate missing data - review impact on analysis")
        
        # Distribution insights
        normal_count = sum(1 for stats in distributions.values() 
                          if 'is_normal' in stats and stats['is_normal'])
        total_numeric = len(distributions)
        
        if total_numeric > 0:
            normal_percentage = (normal_count / total_numeric) * 100
            if normal_percentage > 70:
                insights.append("Most numeric variables follow normal distribution - parametric tests appropriate")
            elif normal_percentage > 30:
                insights.append("Mixed distribution types - consider both parametric and non-parametric methods")
            else:
                insights.append("Mostly non-normal distributions - non-parametric methods recommended")
        
        return insights
    
    def generate_enhanced_ai_insights(self, context: str = "") -> Dict[str, Any]:
        """Generate enhanced AI-powered business insights with improved prompting"""
        if not self.llm_client:
            return {"error": "LLM client not configured. Please provide a valid OpenAI API key and enable AI insights."}
        
        # Prepare enhanced context
        data_summary = self._prepare_enhanced_ai_context()
        
        # Enhanced prompt engineering
        prompt = f"""
        As a Chief Data Officer with 15+ years experience at Fortune 500 companies, provide strategic insights for this dataset.

        DATA OVERVIEW:
        {data_summary}

        BUSINESS CONTEXT:
        {context}

        Provide a comprehensive analysis with these EXACT sections in valid JSON format:

        1. "executive_summary": "Brief overview of key findings and business implications (2-3 sentences)",
        2. "business_opportunities": ["3-5 specific opportunities for growth or improvement"],
        3. "critical_risks": ["3-5 potential risks or issues requiring attention"], 
        4. "strategic_recommendations": ["3-5 actionable recommendations with expected impact"],
        5. "quick_wins": ["2-3 immediate actions with fast ROI"],
        6. "analytics_roadmap": ["4-6 phased steps for data maturity progression"],
        7. "data_quality_insights": ["2-3 insights about data quality and reliability"]

        Focus on:
        - Actionable business insights, not technical details
        - ROI and business value quantification where possible
        - Risk mitigation strategies
        - Scalable solutions for long-term success
        - Industry best practices

        Be specific, data-driven, and business-focused. Avoid generic advice.
        """
        
        try:
            start_time = datetime.now()
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"AI analysis completed in {processing_time:.2f} seconds")
            
            content = response.choices[0].message.content.strip()
            
            # Enhanced response parsing
            try:
                insights = json.loads(content)
                
                # Validate response structure
                required_keys = [
                    "executive_summary", "business_opportunities", "critical_risks",
                    "strategic_recommendations", "quick_wins", "analytics_roadmap",
                    "data_quality_insights"
                ]
                
                for key in required_keys:
                    if key not in insights:
                        insights[key] = [f"No {key.replace('_', ' ')} identified"]
                
                # Cache successful insights
                self.insight_cache[context] = {
                    'insights': insights,
                    'timestamp': datetime.now(),
                    'processing_time': processing_time
                }
                
                return insights
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                return {"error": f"Failed to parse AI response: {e}"}
                
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return {"error": f"API error: {e}"}
        except Exception as e:
            logger.error(f"Unexpected error in AI analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _prepare_enhanced_ai_context(self) -> str:
        """Prepare comprehensive enhanced context for AI analysis"""
        basic_stats = self.get_basic_stats()
        distributions = self.analyze_distributions()
        correlations = self.get_strong_correlations(threshold=0.5)
        categorical_analysis = self.analyze_categorical()
        data_quality_score = self.calculate_data_quality_score()
        
        # Enhanced distribution insights
        dist_insights = []
        for col, stats in list(distributions.items())[:6]:
            if 'error' not in stats:
                # Enhanced distribution characterization
                if stats.get('is_normal', False):
                    dist_type = "normal distribution"
                elif stats['skewness'] > 1:
                    dist_type = "highly right-skewed"
                elif stats['skewness'] < -1:
                    dist_type = "highly left-skewed"
                elif abs(stats['skewness']) > 0.5:
                    dist_type = "moderately skewed"
                else:
                    dist_type = "relatively symmetric"
                
                outlier_info = f", {stats['outliers']} outliers" if stats['outliers'] > 0 else ", no outliers"
                dist_insights.append(f"{col}: {dist_type}{outlier_info}")
        
        # Enhanced correlation insights
        corr_insights = []
        for _, row in correlations.head(4).iterrows():
            direction = "positive" if row['correlation'] > 0 else "negative"
            corr_insights.append(f"{row['variable_1']}-{row['variable_2']}: {direction} {row['strength']} (r={row['correlation']:.2f})")
        
        # Enhanced categorical insights
        cat_insights = []
        for col, stats in list(categorical_analysis.items())[:4]:
            business_context = stats.get('business_relevance', 'General')
            cat_insights.append(f"{col}: {stats['unique_count']} categories, {business_context.lower()}")
        
        # Data quality insights
        quality_insights = []
        if data_quality_score['overall_score'] < 6:
            quality_insights.append("Poor overall data quality - needs improvement")
        elif data_quality_score['overall_score'] < 8:
            quality_insights.append("Moderate data quality - some areas need attention")
        else:
            quality_insights.append("Good data quality - reliable for analysis")
        
        if data_quality_score['completeness'] < 7:
            quality_insights.append(f"Completeness score low ({data_quality_score['completeness']}/10)")
        if data_quality_score['accuracy'] < 7:
            quality_insights.append(f"Accuracy concerns ({data_quality_score['accuracy']}/10)")
        
        context = f"""
        ENHANCED DATASET INTELLIGENCE:
        
        DATASET CHARACTERISTICS:
        - Size: {basic_stats['dataset_shape'][0]:,} rows × {basic_stats['dataset_shape'][1]:,} columns
        - Memory: {basic_stats['memory_usage_mb']:.1f} MB
        - Data Quality Score: {data_quality_score['overall_score']}/10
        - Missing Data: {basic_stats['missing_values']:,} values ({basic_stats['missing_percentage']:.1f}%)
        - Duplicates: {basic_stats['duplicate_rows']:,} rows
        
        DATA COMPOSITION:
        - Numeric Columns: {len(self.numeric_cols)}
        - Categorical Columns: {len(self.categorical_cols)}
        - Date Columns: {len(self.date_cols)}
        
        STATISTICAL PROFILE:
        - Key Distributions: {'; '.join(dist_insights)}
        - Strong Correlations: {'; '.join(corr_insights) if corr_insights else 'None above 0.5 threshold'}
        - Categorical Analysis: {'; '.join(cat_insights)}
        
        DATA QUALITY ASSESSMENT:
        - {'; '.join(quality_insights)}
        - Completeness: {data_quality_score['completeness']}/10
        - Accuracy: {data_quality_score['accuracy']}/10  
        - Consistency: {data_quality_score['consistency']}/10
        - Timeliness: {data_quality_score['timeliness']}/10
        - Validity: {data_quality_score['validity']}/10
        
        COLUMN SAMPLES:
        - Numeric: {', '.join(self.numeric_cols[:8])}{'...' if len(self.numeric_cols) > 8 else ''}
        - Categorical: {', '.join(self.categorical_cols[:8])}{'...' if len(self.categorical_cols) > 8 else ''}
        """
        
        return context
    
    def create_qq_plot(self, col: str) -> Optional[go.Figure]:
        """Create enhanced Q-Q plot for normality check"""
        data = self.df[col].dropna()
        
        if len(data) < 2:
            return None
        
        try:
            # Create Q-Q plot
            qq = stats.probplot(data, dist="norm")
            x = qq[0][0]
            y = qq[0][1]
            slope = qq[1][0]
            intercept = qq[1][1]
            r_squared = qq[1][2] ** 2
            
            fig = go.Figure()
            
            # Data points
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Data Points',
                marker=dict(color='blue', size=6, opacity=0.6),
                hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
            ))
            
            # Normal line
            fig.add_trace(go.Scatter(
                x=x, y=slope * x + intercept,
                mode='lines',
                name=f'Normal Line (R²={r_squared:.3f})',
                line=dict(color='red', dash='dash', width=2)
            ))
            
            # Enhanced layout
            fig.update_layout(
                title=f'Q-Q Plot for {col} - Normality Check',
                xaxis_title='Theoretical Quantiles',
                yaxis_title='Sample Quantiles',
                showlegend=True,
                template='plotly_white',
                annotations=[
                    dict(
                        x=0.05, y=0.95, xref='paper', yref='paper',
                        text=f'R² = {r_squared:.3f}',
                        showarrow=False,
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=1
                    )
                ]
            )
            
            return fig
        except Exception as e:
            logger.error(f"Error creating Q-Q plot for {col}: {e}")
            return None
    
    def generate_insights(self) -> Dict[str, List[str]]:
        """Generate enhanced automated EDA insights (fallback without AI)"""
        insights = {
            'data_quality': [],
            'distributions': [],
            'relationships': [],
            'anomalies': [],
            'recommendations': [],
            'business_implications': []
        }
        
        basic_stats = self.get_basic_stats()
        distributions = self.analyze_distributions()
        correlations = self.get_strong_correlations(threshold=0.7)
        data_quality_score = self.calculate_data_quality_score()
        
        # Enhanced data quality insights
        insights['data_quality'].append(
            f"Dataset has {basic_stats['dataset_shape'][0]:,} rows and {basic_stats['dataset_shape'][1]:,} columns"
        )
        
        if data_quality_score['overall_score'] < 7:
            insights['data_quality'].append(
                f"Data quality needs improvement (score: {data_quality_score['overall_score']}/10)"
            )
        
        if basic_stats['missing_values'] > 0:
            insights['data_quality'].append(
                f"Found {basic_stats['missing_values']:,} missing values ({basic_stats['missing_percentage']:.1f}% of data)"
            )
        
        if basic_stats['duplicate_rows'] > 0:
            insights['data_quality'].append(
                f"Found {basic_stats['duplicate_rows']:,} duplicate rows that should be reviewed"
            )
        
        # Enhanced distribution insights
        normal_vars = []
        skewed_vars = []
        
        for col, stats in distributions.items():
            if 'error' in stats:
                continue
                
            if stats.get('is_normal', False):
                normal_vars.append(col)
            elif abs(stats['skewness']) > 1:
                skewed_vars.append((col, stats['skewness']))
            
            if stats['outliers'] > len(self.df) * 0.05:
                insights['distributions'].append(
                    f"Column '{col}' has many outliers ({stats['outliers']}) that may need treatment"
                )
        
        if normal_vars:
            insights['distributions'].append(
                f"{len(normal_vars)} variables follow normal distribution: {', '.join(normal_vars[:3])}"
            )
        
        if skewed_vars:
            top_skewed = sorted(skewed_vars, key=lambda x: abs(x[1]), reverse=True)[:3]
            insights['distributions'].append(
                f"Highly skewed variables: {', '.join([f'{col} (skew: {skew:.1f})' for col, skew in top_skewed])}"
            )
        
        # Enhanced relationship insights
        if not correlations.empty:
            strong_corrs = [c for c in correlations.to_dict('records') if c['strength'] in ['strong', 'very strong']]
            if strong_corrs:
                insights['relationships'].append(
                    f"Found {len(strong_corrs)} strong correlations that may indicate important relationships"
                )
                for corr in strong_corrs[:3]:
                    insights['relationships'].append(
                        f"{corr['variable_1']} ↔ {corr['variable_2']}: {corr['correlation']:.2f} ({corr['strength']})"
                    )
        
        # Enhanced anomaly insights
        anomalies = self.detect_anomalies()
        if not anomalies.empty:
            anomaly_cols = anomalies['anomaly_column'].unique()
            high_severity = anomalies[anomalies.get('severity', '') == 'high']
            
            insights['anomalies'].append(
                f"Detected {len(anomalies)} anomalies across {len(anomaly_cols)} columns"
            )
            
            if len(high_severity) > 0:
                insights['anomalies'].append(
                    f"{len(high_severity)} high-severity anomalies require immediate attention"
                )
        
        # Enhanced business implications
        if len(self.numeric_cols) >= 3:
            insights['business_implications'].append(
                "Multiple numeric variables available for predictive modeling and trend analysis"
            )
        
        if len(self.categorical_cols) >= 2:
            insights['business_implications'].append(
                "Categorical variables enable segmentation and cohort analysis"
            )
        
        if not correlations.empty:
            insights['business_implications'].append(
                "Strong correlations suggest potential causal relationships to investigate"
            )
        
        # Enhanced recommendations
        if basic_stats['missing_values'] > 0:
            insights['recommendations'].append(
                "Implement data validation rules and imputation strategies for missing values"
            )
        
        if any(stats.get('skewness', 0) > 2 for stats in distributions.values()):
            insights['recommendations'].append(
                "Apply logarithmic or power transformations to highly skewed variables"
            )
        
        if not correlations.empty:
            insights['recommendations'].append(
                "Investigate strongly correlated variables for business insights and potential multicollinearity"
            )
        
        if len(self.numeric_cols) >= 3:
            insights['recommendations'].append(
                "Consider dimensionality reduction techniques like PCA for better visualization and analysis"
            )
        
        if data_quality_score['overall_score'] < 7:
            insights['recommendations'].append(
                "Establish data quality monitoring and governance processes"
            )
        
        return insights