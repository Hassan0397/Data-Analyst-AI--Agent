import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, normaltest, anderson, levene, bartlett
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
import math
from dataclasses import dataclass
import json
from datetime import datetime
import base64

try:
    from statsmodels.stats.power import TTestPower, FTestAnovaPower, GofChisquarePower
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    from statsmodels.multivariate.manova import MANOVA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from factor_analyzer import FactorAnalyzer
    HAS_ADVANCED_STATS = True
except ImportError:
    HAS_ADVANCED_STATS = False

@dataclass
class TestAssumptions:
    """Data class for test assumption results"""
    normality: bool
    equal_variance: bool
    independence: bool
    sample_size: bool
    outliers: bool
    is_met: bool
    warnings: List[str]

@dataclass
class BayesianResult:
    """Data class for Bayesian test results"""
    bayes_factor: float
    evidence_strength: str
    prior: float
    posterior: float
    interpretation: str

class AdvancedStatisticalTests:
    """Advanced statistical testing utilities with comprehensive diagnostics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._effect_sizes_cache = {}
        self._analysis_history = []
    
    def _log_analysis(self, test_type: str, parameters: Dict, results: Dict):
        """Log analysis for history and export"""
        self._analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'test_type': test_type,
            'parameters': parameters,
            'results': {k: v for k, v in results.items() if k != 'raw_data'}
        })
    
    def export_analysis_history(self, format: str = 'json') -> Union[str, bytes]:
        """Export analysis history"""
        if format == 'json':
            return json.dumps(self._analysis_history, indent=2)
        elif format == 'csv':
            # Create a flattened version for CSV
            records = []
            for analysis in self._analysis_history:
                record = {
                    'timestamp': analysis['timestamp'],
                    'test_type': analysis['test_type']
                }
                # Flatten parameters and results
                for key, value in analysis['parameters'].items():
                    if isinstance(value, (str, int, float, bool)):
                        record[f'param_{key}'] = value
                
                for key, value in analysis['results'].items():
                    if isinstance(value, (str, int, float, bool)):
                        record[f'result_{key}'] = value
                
                records.append(record)
            
            df = pd.DataFrame(records)
            return df.to_csv(index=False)
        else:
            raise ValueError("Unsupported format. Use 'json' or 'csv'")
    
    def _detect_outliers(self, data: pd.Series, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers using multiple methods"""
        if len(data) < 5:
            return {'outliers_count': 0, 'outlier_indices': [], 'method': method}
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
        else:
            outliers = pd.Series([], dtype=data.dtype)
        
        return {
            'outliers_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'outlier_indices': outliers.index.tolist(),
            'method': method
        }
    
    def _calculate_effect_size(self, test_type: str, **kwargs) -> Dict[str, Any]:
        """Calculate effect sizes for various statistical tests"""
        cache_key = f"{test_type}_{hash(str(kwargs))}"
        if cache_key in self._effect_sizes_cache:
            return self._effect_sizes_cache[cache_key]
        
        effect_sizes = {}
        
        try:
            if test_type == 'ttest_one_sample':
                data = kwargs['data']
                test_mean = kwargs['test_mean']
                cohens_d = abs(data.mean() - test_mean) / data.std()
                effect_sizes = {
                    'cohens_d': cohens_d,
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
            
            elif test_type == 'ttest_independent':
                data1 = kwargs['data1']
                data2 = kwargs['data2']
                pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
                cohens_d = abs(data1.mean() - data2.mean()) / pooled_std
                effect_sizes = {
                    'cohens_d': cohens_d,
                    'hedges_g': cohens_d * (1 - (3/(4*(len(data1)+len(data2)-2)-1))),
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
            
            elif test_type == 'anova':
                groups = kwargs['groups']
                overall_mean = np.concatenate(groups).mean()
                ss_between = sum(len(g) * (g.mean() - overall_mean)**2 for g in groups)
                ss_total = sum(np.sum((g - overall_mean)**2 for g in groups))
                eta_squared = ss_between / ss_total
                
                effect_sizes = {
                    'eta_squared': eta_squared,
                    'interpretation': self._interpret_eta_squared(eta_squared)
                }
            
            elif test_type == 'chi_square':
                contingency_table = kwargs['contingency_table']
                chi2 = kwargs['chi2_statistic']
                n = contingency_table.sum().sum()
                min_dim = min(contingency_table.shape) - 1
                
                phi = np.sqrt(chi2 / n)
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                
                effect_sizes = {
                    'phi': phi,
                    'cramers_v': cramers_v,
                    'interpretation': self._interpret_cramers_v(cramers_v)
                }
        
        except Exception as e:
            effect_sizes = {'error': f'Effect size calculation failed: {str(e)}'}
        
        self._effect_sizes_cache[cache_key] = effect_sizes
        return effect_sizes
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return "Very Small"
        elif abs(d) < 0.5:
            return "Small"
        elif abs(d) < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta squared effect size"""
        if eta2 < 0.01:
            return "Very Small"
        elif eta2 < 0.06:
            return "Small"
        elif eta2 < 0.14:
            return "Medium"
        else:
            return "Large"
    
    def _interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramer's V effect size"""
        if v < 0.1:
            return "Very Small"
        elif v < 0.3:
            return "Small"
        elif v < 0.5:
            return "Medium"
        else:
            return "Large"
    
    def check_test_assumptions(self, test_type: str, **kwargs) -> TestAssumptions:
        """Comprehensive assumption checking for statistical tests"""
        warnings_list = []
        normality = True
        equal_variance = True
        independence = True  # Assumed true, requires experimental design knowledge
        sample_size = True
        outliers = True
        
        try:
            if test_type in ['ttest_one_sample', 'ttest_paired']:
                data = kwargs.get('data')
                if data is not None:
                    # Normality
                    if len(data) >= 3 and len(data) <= 5000:
                        _, normality_p = shapiro(data)
                        if normality_p < 0.05:
                            normality = False
                            warnings_list.append("Data may not be normally distributed")
                    
                    # Outliers
                    outlier_result = self._detect_outliers(data)
                    if outlier_result['outlier_percentage'] > 5:
                        outliers = False
                        warnings_list.append(f"High percentage of outliers detected ({outlier_result['outlier_percentage']:.1f}%)")
                    
                    # Sample size
                    if len(data) < 30:
                        sample_size = False
                        warnings_list.append("Small sample size may affect test power")
            
            elif test_type == 'ttest_independent':
                data1 = kwargs.get('data1')
                data2 = kwargs.get('data2')
                
                if data1 is not None and data2 is not None:
                    # Normality
                    for i, data in enumerate([data1, data2], 1):
                        if len(data) >= 3 and len(data) <= 5000:
                            _, normality_p = shapiro(data)
                            if normality_p < 0.05:
                                normality = False
                                warnings_list.append(f"Group {i} may not be normally distributed")
                    
                    # Equal variance
                    _, levene_p = levene(data1, data2)
                    if levene_p < 0.05:
                        equal_variance = False
                        warnings_list.append("Groups may have unequal variances")
                    
                    # Sample size
                    if len(data1) < 20 or len(data2) < 20:
                        sample_size = False
                        warnings_list.append("Small sample sizes may affect test power")
            
            elif test_type == 'anova':
                groups = kwargs.get('groups', [])
                
                # Normality for each group
                for i, group in enumerate(groups):
                    if len(group) >= 3 and len(group) <= 5000:
                        _, normality_p = shapiro(group)
                        if normality_p < 0.05:
                            normality = False
                            warnings_list.append(f"Group {i+1} may not be normally distributed")
                
                # Homogeneity of variance
                if len(groups) >= 2:
                    _, bartlett_p = bartlett(*groups)
                    if bartlett_p < 0.05:
                        equal_variance = False
                        warnings_list.append("Groups may have unequal variances")
                
                # Sample size
                if any(len(group) < 20 for group in groups):
                    sample_size = False
                    warnings_list.append("Some groups have small sample sizes")
        
        except Exception as e:
            warnings_list.append(f"Assumption checking error: {str(e)}")
        
        is_met = all([normality, equal_variance, independence, sample_size, outliers])
        
        return TestAssumptions(
            normality=normality,
            equal_variance=equal_variance,
            independence=independence,
            sample_size=sample_size,
            outliers=outliers,
            is_met=is_met,
            warnings=warnings_list
        )
    
    # Bayesian Statistical Tests
    def bayesian_ttest(self, col1: str, col2: str = None, test_mean: float = 0, 
                      paired: bool = False, prior_scale: float = 0.707) -> Dict[str, Any]:
        """Bayesian t-test implementation"""
        try:
            if not HAS_ADVANCED_STATS:
                return {'error': 'Bayesian tests require additional dependencies: pip install pymc3 arviz'}
            
            # Simplified Bayesian analysis (approximation)
            if paired and col2 is not None:
                data1 = self.df[col1].dropna()
                data2 = self.df[col2].dropna()
                paired_data = pd.DataFrame({col1: data1, col2: data2}).dropna()
                differences = paired_data[col2] - paired_data[col1]
                effect_size = differences.mean() / differences.std()
            elif col2 is not None:
                # Two sample
                data1 = self.df[col1].dropna()
                data2 = self.df[col2].dropna()
                pooled_std = np.sqrt(((len(data1)-1)*data1.std()**2 + (len(data2)-1)*data2.std()**2) / (len(data1)+len(data2)-2))
                effect_size = (data1.mean() - data2.mean()) / pooled_std
            else:
                # One sample
                data = self.df[col1].dropna()
                effect_size = (data.mean() - test_mean) / data.std()
            
            # Approximate Bayes factor
            bayes_factor = np.exp(abs(effect_size) * prior_scale)
            
            # Interpret Bayes factor
            if bayes_factor > 100:
                evidence = "Decisive evidence for H1"
            elif bayes_factor > 30:
                evidence = "Very strong evidence for H1"
            elif bayes_factor > 10:
                evidence = "Strong evidence for H1"
            elif bayes_factor > 3:
                evidence = "Moderate evidence for H1"
            elif bayes_factor > 1:
                evidence = "Anecdotal evidence for H1"
            elif bayes_factor > 1/3:
                evidence = "Anecdotal evidence for H0"
            elif bayes_factor > 1/10:
                evidence = "Moderate evidence for H0"
            else:
                evidence = "Strong evidence for H0"
            
            return {
                'bayes_factor': float(bayes_factor),
                'evidence_strength': evidence,
                'interpretation': evidence,
                'test_type': 'bayesian_ttest',
                'paired': paired,
                'effect_size': float(effect_size)
            }
            
        except Exception as e:
            return {'error': f'Bayesian t-test failed: {str(e)}'}
    
    # Time Series Analysis
    def time_series_analysis(self, column: str, date_column: str = None, 
                           frequency: str = 'D') -> Dict[str, Any]:
        """Comprehensive time series analysis"""
        try:
            if date_column and date_column in self.df.columns:
                ts_data = self.df.set_index(pd.to_datetime(self.df[date_column]))[column]
            else:
                ts_data = pd.Series(self.df[column].values, 
                                  index=pd.date_range(start='2020-01-01', periods=len(self.df), freq=frequency))
            
            # Stationarity test
            adf_result = adfuller(ts_data.dropna())
            
            # Basic decomposition (simplified)
            if len(ts_data) > 12:
                try:
                    decomposition = seasonal_decompose(ts_data, period=min(12, len(ts_data)//2), 
                                                    extrapolate_trend='freq')
                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    residual = decomposition.resid
                except:
                    trend = ts_data.rolling(window=5, center=True).mean()
                    seasonal = None
                    residual = ts_data - trend
            else:
                trend = ts_data.rolling(window=3, center=True).mean()
                seasonal = None
                residual = ts_data - trend
            
            # ACF/PACF
            acf_values = acf(ts_data.dropna(), nlags=min(20, len(ts_data)//4))
            pacf_values = pacf(ts_data.dropna(), nlags=min(20, len(ts_data)//4))
            
            return {
                'stationary': adf_result[1] < 0.05,
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'acf': acf_values,
                'pacf': pacf_values,
                'test_type': 'time_series_analysis'
            }
            
        except Exception as e:
            return {'error': f'Time series analysis failed: {str(e)}'}
    
    # Survival Analysis
    def survival_analysis(self, time_column: str, event_column: str, 
                         group_column: str = None) -> Dict[str, Any]:
        """Survival analysis with Kaplan-Meier and Cox regression"""
        try:
            if not HAS_ADVANCED_STATS:
                return {'error': 'Survival analysis requires lifelines: pip install lifelines'}
            
            data = self.df[[time_column, event_column]].dropna()
            
            # Kaplan-Meier
            kmf = KaplanMeierFitter()
            kmf.fit(data[time_column], data[event_column])
            
            # Cox regression if group column provided
            cox_results = None
            if group_column and group_column in self.df.columns:
                cox_data = self.df[[time_column, event_column, group_column]].dropna()
                cox_data[group_column] = cox_data[group_column].astype('category').cat.codes
                
                cph = CoxPHFitter()
                cph.fit(cox_data, duration_col=time_column, event_col=event_column)
                cox_results = {
                    'hazard_ratios': cph.hazard_ratios_,
                    'summary': cph.summary
                }
            
            return {
                'km_curve': kmf.survival_function_,
                'median_survival': kmf.median_survival_time_,
                'cox_results': cox_results,
                'test_type': 'survival_analysis'
            }
            
        except Exception as e:
            return {'error': f'Survival analysis failed: {str(e)}'}
    
    # Multivariate Analysis
    def manova_analysis(self, numeric_cols: List[str], group_col: str) -> Dict[str, Any]:
        """MANOVA analysis for multiple dependent variables"""
        try:
            if not HAS_ADVANCED_STATS:
                return {'error': 'MANOVA requires statsmodels'}
            
            # Check if we have enough data
            if len(numeric_cols) < 2:
                return {'error': 'MANOVA requires at least 2 dependent variables'}
            
            # Prepare data
            manova_data = self.df[numeric_cols + [group_col]].dropna()
            
            if len(manova_data) < len(numeric_cols) + 1:
                return {'error': 'Insufficient data for MANOVA'}
            
            formula = f"{' + '.join(numeric_cols)} ~ {group_col}"
            manova = MANOVA.from_formula(formula, data=manova_data)
            result = manova.mv_test()
            
            return {
                'manova_results': result.summary(),
                'test_type': 'manova',
                'dependent_vars': numeric_cols,
                'independent_var': group_col
            }
            
        except Exception as e:
            return {'error': f'MANOVA analysis failed: {str(e)}'}
    
    def factor_analysis(self, numeric_cols: List[str], n_factors: int = None) -> Dict[str, Any]:
        """Exploratory Factor Analysis"""
        try:
            if not HAS_ADVANCED_STATS:
                return {'error': 'Factor analysis requires factor_analyzer: pip install factor-analyzer'}
            
            data = self.df[numeric_cols].dropna()
            
            if n_factors is None:
                n_factors = min(5, len(numeric_cols) // 2)
            
            fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
            fa.fit(data)
            
            return {
                'loadings': fa.loadings_,
                'variance_explained': fa.get_factor_variance(),
                'communalities': fa.get_communalities(),
                'test_type': 'factor_analysis',
                'n_factors': n_factors
            }
            
        except Exception as e:
            return {'error': f'Factor analysis failed: {str(e)}'}
    
    # Batch Processing
    def batch_normality_tests(self, columns: List[str]) -> Dict[str, Any]:
        """Perform normality tests on multiple columns"""
        results = {}
        for col in columns:
            if col in self.numeric_cols:
                results[col] = self.perform_normality_tests(col)
        return results
    
    def batch_correlation_analysis(self, columns: List[str]) -> Dict[str, Any]:
        """Perform correlation analysis on multiple columns"""
        if len(columns) < 2:
            return {'error': 'Need at least 2 columns for correlation analysis'}
        
        corr_matrix = self.df[columns].corr()
        return {
            'correlation_matrix': corr_matrix,
            'strong_correlations': corr_matrix[(corr_matrix.abs() > 0.7) & (corr_matrix.abs() < 1.0)]
        }

    # Core Statistical Tests (Fixed and Enhanced)
    def perform_normality_tests(self, column: str) -> Dict[str, Any]:
        """Enhanced normality testing with comprehensive diagnostics"""
        data = self.df[column].dropna()
        
        if len(data) < 3:
            return {'error': 'Insufficient data for normality tests (need at least 3 observations)'}
        
        results = {}
        
        # Basic statistics
        results['descriptive_stats'] = {
            'n': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        # Outlier analysis
        results['outliers'] = self._detect_outliers(data)
        
        # Shapiro-Wilk test
        if len(data) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(data)
                results['shapiro'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05,
                    'test': 'Shapiro-Wilk',
                    'interpretation': 'Normal' if shapiro_p > 0.05 else 'Not Normal'
                }
            except Exception as e:
                results['shapiro'] = {'error': str(e)}
        
        # D'Agostino's test
        try:
            dagostino_stat, dagostino_p = normaltest(data)
            results['dagostino'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > 0.05,
                'test': "D'Agostino",
                'interpretation': 'Normal' if dagostino_p > 0.05 else 'Not Normal'
            }
        except Exception as e:
            results['dagostino'] = {'error': str(e)}
        
        # Anderson-Darling test
        try:
            anderson_result = anderson(data, dist='norm')
            results['anderson'] = {
                'statistic': anderson_result.statistic,
                'critical_value': anderson_result.critical_values[2],
                'is_normal': anderson_result.statistic < anderson_result.critical_values[2],
                'test': 'Anderson-Darling',
                'interpretation': 'Normal' if anderson_result.statistic < anderson_result.critical_values[2] else 'Not Normal'
            }
        except Exception as e:
            results['anderson'] = {'error': str(e)}
        
        # Overall normality assessment
        normal_tests = [r for r in results.values() if isinstance(r, dict) and 'is_normal' in r and r['is_normal'] is not None]
        if normal_tests:
            results['overall_normality'] = {
                'is_normal': sum(test['is_normal'] for test in normal_tests) / len(normal_tests) > 0.5,
                'tests_agree': len(set(test['is_normal'] for test in normal_tests)) == 1,
                'tests_performed': len(normal_tests)
            }
        
        return results

    def one_sample_ttest(self, column: str, test_mean: float, alternative: str = 'two-sided', alpha: float = 0.05) -> Dict[str, Any]:
        """Enhanced one-sample t-test with diagnostics"""
        data = self.df[column].dropna()
        
        if len(data) < 2:
            return {'error': 'Insufficient data for t-test (need at least 2 observations)'}
        
        try:
            # Check assumptions
            assumptions = self.check_test_assumptions('ttest_one_sample', data=data)
            
            # Perform t-test
            t_statistic, p_value = stats.ttest_1samp(data, test_mean, alternative=alternative)
            
            # Calculate confidence interval
            confidence_interval = stats.t.interval(
                1 - alpha, len(data) - 1, loc=data.mean(), scale=stats.sem(data)
            )
            
            # Calculate effect size
            effect_sizes = self._calculate_effect_size('ttest_one_sample', data=data, test_mean=test_mean)
            
            result = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'df': len(data) - 1,
                'sample_mean': data.mean(),
                'test_mean': test_mean,
                'confidence_interval': confidence_interval,
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'one_sample_ttest',
                'assumptions': assumptions,
                'effect_sizes': effect_sizes,
                'sample_size': len(data),
                'standard_error': stats.sem(data)
            }
            
            self._log_analysis('one_sample_ttest', 
                             {'column': column, 'test_mean': test_mean, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'T-test failed: {str(e)}'}

    def two_sample_ttest(self, col1: str, col2: str, alternative: str = 'two-sided',
                        equal_var: bool = None, alpha: float = 0.05) -> Dict[str, Any]:
        """Enhanced two-sample t-test with automatic variance checking"""
        data1 = self.df[col1].dropna()
        data2 = self.df[col2].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return {'error': 'Insufficient data for two-sample t-test'}
        
        try:
            # Automatically check equal variance if not specified
            if equal_var is None:
                _, levene_p = levene(data1, data2)
                equal_var = levene_p > 0.05
            
            # Check assumptions
            assumptions = self.check_test_assumptions('ttest_independent', data1=data1, data2=data2)
            
            # Perform t-test
            t_statistic, p_value = stats.ttest_ind(data1, data2, alternative=alternative, equal_var=equal_var)
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_size('ttest_independent', data1=data1, data2=data2)
            
            result = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'df': len(data1) + len(data2) - 2,
                'mean1': data1.mean(),
                'mean2': data2.mean(),
                'mean_difference': data1.mean() - data2.mean(),
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'two_sample_ttest',
                'equal_variance': equal_var,
                'assumptions': assumptions,
                'effect_sizes': effect_sizes,
                'sample_sizes': [len(data1), len(data2)],
                'standard_errors': [stats.sem(data1), stats.sem(data2)]
            }
            
            self._log_analysis('two_sample_ttest', 
                             {'col1': col1, 'col2': col2, 'alpha': alpha, 'equal_var': equal_var}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'Two-sample t-test failed: {str(e)}'}

    def paired_ttest(self, pre_col: str, post_col: str, alternative: str = 'two-sided', alpha: float = 0.05) -> Dict[str, Any]:
        """Enhanced paired t-test with diagnostics"""
        # Remove rows where either value is missing
        paired_data = self.df[[pre_col, post_col]].dropna()
        
        if len(paired_data) < 2:
            return {'error': 'Insufficient data for paired t-test (need at least 2 complete pairs)'}
        
        try:
            pre_data = paired_data[pre_col]
            post_data = paired_data[post_col]
            differences = post_data - pre_data
            
            # Check assumptions
            assumptions = self.check_test_assumptions('ttest_paired', data=differences)
            
            # Perform paired t-test
            t_statistic, p_value = stats.ttest_rel(pre_data, post_data, alternative=alternative)
            
            # Calculate confidence interval for mean difference
            confidence_interval = stats.t.interval(
                1 - alpha, len(differences) - 1, loc=differences.mean(), scale=stats.sem(differences)
            )
            
            # Calculate effect size
            effect_sizes = self._calculate_effect_size('ttest_one_sample', data=differences, test_mean=0)
            
            result = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'df': len(differences) - 1,
                'mean_difference': differences.mean(),
                'confidence_interval': confidence_interval,
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'paired_ttest',
                'assumptions': assumptions,
                'effect_sizes': effect_sizes,
                'sample_size': len(differences),
                'pre_mean': pre_data.mean(),
                'post_mean': post_data.mean()
            }
            
            self._log_analysis('paired_ttest', 
                             {'pre_col': pre_col, 'post_col': post_col, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'Paired t-test failed: {str(e)}'}

    def one_way_anova(self, numeric_col: str, group_col: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Enhanced ANOVA with post-hoc power and effect sizes"""
        groups = []
        group_data = {}
        
        for group_name, group_df in self.df.groupby(group_col)[numeric_col]:
            clean_data = group_df.dropna()
            if len(clean_data) >= 1:
                groups.append(clean_data)
                group_data[group_name] = clean_data
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for ANOVA'}
        
        try:
            # Check assumptions
            assumptions = self.check_test_assumptions('anova', groups=groups)
            
            # Perform ANOVA
            f_statistic, p_value = stats.f_oneway(*groups)
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_size('anova', groups=groups)
            
            # Group statistics
            group_stats = {}
            for name, data in group_data.items():
                group_stats[name] = {
                    'n': len(data),
                    'mean': data.mean(),
                    'std': data.std(),
                    'se': stats.sem(data)
                }
            
            result = {
                'f_statistic': f_statistic,
                'p_value': p_value,
                'n_groups': len(groups),
                'group_names': list(group_data.keys()),
                'group_stats': group_stats,
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'one_way_anova',
                'assumptions': assumptions,
                'effect_sizes': effect_sizes,
                'total_observations': sum(len(g) for g in groups)
            }
            
            self._log_analysis('one_way_anova', 
                             {'numeric_col': numeric_col, 'group_col': group_col, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'ANOVA failed: {str(e)}'}

    def tukey_hsd(self, numeric_col: str, group_col: str) -> pd.DataFrame:
        """Perform Tukey HSD post-hoc test"""
        try:
            # Prepare data for Tukey HSD
            data = []
            groups_list = []
            
            for group_name, group_data in self.df.groupby(group_col)[numeric_col]:
                valid_data = group_data.dropna()
                if len(valid_data) >= 1:
                    data.extend(valid_data)
                    groups_list.extend([group_name] * len(valid_data))
            
            if len(set(groups_list)) < 2:
                return pd.DataFrame({'error': ['Insufficient groups for Tukey HSD']})
            
            # Perform Tukey HSD
            tukey_result = pairwise_tukeyhsd(data, groups_list, alpha=0.05)
            
            # Convert to DataFrame
            result_df = pd.DataFrame({
                'group1': tukey_result.groupsunique[tukey_result._results[0][0]],
                'group2': tukey_result.groupsunique[tukey_result._results[0][1]],
                'mean_diff': tukey_result.meandiffs,
                'p_adj': tukey_result.pvalues,
                'reject': tukey_result.reject,
                'lower_ci': tukey_result.confint[:, 0],
                'upper_ci': tukey_result.confint[:, 1]
            })
            
            return result_df
        except ImportError:
            return pd.DataFrame({'error': ['statsmodels not installed. Install with: pip install statsmodels']})
        except Exception as e:
            return pd.DataFrame({'error': [f'Tukey HSD failed: {str(e)}']})

    def chi_square_test(self, col1: str, col2: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Enhanced chi-square test with diagnostics"""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(self.df[col1], self.df[col2])
            
            if contingency_table.size == 0:
                return {'error': 'Contingency table is empty'}
            
            # Check expected frequencies
            n = contingency_table.sum().sum()
            expected = stats.contingency.expected_freq(contingency_table)
            low_expected = (expected < 5).sum()
            
            if low_expected > contingency_table.size * 0.2:
                return {'error': 'Too many expected counts < 5 for reliable chi-square test'}
            
            # Perform chi-square test
            chi2_statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_size(
                'chi_square', 
                contingency_table=contingency_table,
                chi2_statistic=chi2_statistic
            )
            
            # Calculate residuals
            residuals = (contingency_table - expected) / np.sqrt(expected)
            
            result = {
                'chi2_statistic': chi2_statistic,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'chi_square',
                'contingency_table': contingency_table,
                'expected_frequencies': expected,
                'standardized_residuals': residuals,
                'effect_sizes': effect_sizes,
                'total_observations': n,
                'low_expected_warning': low_expected > 0
            }
            
            self._log_analysis('chi_square_test', 
                             {'col1': col1, 'col2': col2, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'Chi-square test failed: {str(e)}'}

    def mann_whitney_u(self, col1: str, col2: str, alternative: str = 'two-sided',
                      alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Mann-Whitney U test (non-parametric two-sample test)"""
        data1 = self.df[col1].dropna()
        data2 = self.df[col2].dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return {'error': 'Insufficient data for Mann-Whitney U test'}
        
        try:
            # Perform Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
            
            result = {
                'statistic': statistic,
                'p_value': p_value,
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'mann_whitney_u',
                'sample_sizes': [len(data1), len(data2)]
            }
            
            self._log_analysis('mann_whitney_u', 
                             {'col1': col1, 'col2': col2, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'Mann-Whitney U test failed: {str(e)}'}

    def kruskal_wallis(self, numeric_col: str, group_col: str, alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Kruskal-Wallis H test (non-parametric ANOVA)"""
        groups = []
        
        for group_name, group_data in self.df.groupby(group_col)[numeric_col]:
            valid_data = group_data.dropna()
            if len(valid_data) >= 1:
                groups.append(valid_data)
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for Kruskal-Wallis test'}
        
        try:
            # Perform Kruskal-Wallis test
            statistic, p_value = stats.kruskal(*groups)
            
            result = {
                'statistic': statistic,
                'p_value': p_value,
                'n_groups': len(groups),
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'kruskal_wallis'
            }
            
            self._log_analysis('kruskal_wallis', 
                             {'numeric_col': numeric_col, 'group_col': group_col, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'Kruskal-Wallis test failed: {str(e)}'}

    def wilcoxon_signed_rank(self, pre_col: str, post_col: str, 
                           alternative: str = 'two-sided', alpha: float = 0.05) -> Dict[str, Any]:
        """Perform Wilcoxon signed-rank test (non-parametric paired test)"""
        # Remove rows where either value is missing
        paired_data = self.df[[pre_col, post_col]].dropna()
        
        if len(paired_data) < 2:
            return {'error': 'Insufficient data for Wilcoxon signed-rank test'}
        
        try:
            pre_data = paired_data[pre_col]
            post_data = paired_data[post_col]
            
            # Perform Wilcoxon test
            statistic, p_value = stats.wilcoxon(pre_data, post_data, alternative=alternative)
            
            result = {
                'statistic': statistic,
                'p_value': p_value,
                'alpha': alpha,
                'reject_null': p_value < alpha,
                'test_type': 'wilcoxon_signed_rank',
                'sample_size': len(paired_data)
            }
            
            self._log_analysis('wilcoxon_signed_rank', 
                             {'pre_col': pre_col, 'post_col': post_col, 'alpha': alpha}, 
                             result)
            return result
        except Exception as e:
            return {'error': f'Wilcoxon signed-rank test failed: {str(e)}'}

    def power_analysis(self, test_type: str, effect_size: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
        """Perform power analysis for sample size determination"""
        try:
            if test_type == 'ttest':
                analysis = TTestPower()
                sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power)
                return {
                    'required_sample_size': int(np.ceil(sample_size)),
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'test_type': test_type
                }
            elif test_type == 'anova':
                analysis = FTestAnovaPower()
                sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, k_groups=2)
                return {
                    'required_sample_size_per_group': int(np.ceil(sample_size)),
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'test_type': test_type
                }
            elif test_type == 'chi2':
                analysis = GofChisquarePower()
                sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, n_bins=2)
                return {
                    'required_sample_size': int(np.ceil(sample_size)),
                    'effect_size': effect_size,
                    'alpha': alpha,
                    'power': power,
                    'test_type': test_type
                }
        except ImportError:
            return {'error': 'statsmodels required for power analysis. Install with: pip install statsmodels'}
        except Exception as e:
            return {'error': f'Power analysis failed: {str(e)}'}

    def generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive statistical summary with insights"""
        summary = {
            'dataset_overview': {},
            'normality_assessment': {},
            'correlation_analysis': {},
            'statistical_power': {},
            'data_quality': {},
            'advanced_insights': {},
            'recommendations': []
        }
        
        # Dataset overview
        summary['dataset_overview'] = {
            'total_observations': len(self.df),
            'numeric_variables': len(self.numeric_cols),
            'categorical_variables': len(self.categorical_cols),
            'missing_data_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_rows': self.df.duplicated().sum()
        }
        
        # Normality assessment for first 3 numeric columns
        for col in self.numeric_cols[:3]:
            normality_results = self.perform_normality_tests(col)
            if 'error' not in normality_results:
                summary['normality_assessment'][col] = {
                    'is_normal': normality_results.get('overall_normality', {}).get('is_normal', False),
                    'skewness': normality_results.get('descriptive_stats', {}).get('skewness', 0),
                    'outliers_percentage': normality_results.get('outliers', {}).get('outlier_percentage', 0)
                }
        
        # Correlation analysis
        if len(self.numeric_cols) >= 2:
            corr_matrix = self.df[self.numeric_cols].corr()
            strong_corrs = (corr_matrix.abs() > 0.7) & (corr_matrix.abs() < 1.0)
            
            summary['correlation_analysis'] = {
                'strong_correlations_count': strong_corrs.sum().sum() // 2,
                'highest_correlation': corr_matrix.abs().max().max(),
                'correlation_matrix': corr_matrix
            }
        
        # Statistical power assessment
        if len(self.numeric_cols) > 0:
            sample_size = len(self.df)
            if sample_size < 30:
                power_status = "Low"
            elif sample_size < 100:
                power_status = "Medium"
            else:
                power_status = "High"
            
            summary['statistical_power'] = {
                'sample_size': sample_size,
                'power_status': power_status,
                'recommended_min_sample': 30
            }
        
        # Data quality assessment
        missing_by_column = self.df.isnull().sum() / len(self.df) * 100
        high_missing_cols = missing_by_column[missing_by_column > 20].index.tolist()
        
        summary['data_quality'] = {
            'high_missing_columns': high_missing_cols,
            'completeness_score': 100 - (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'data_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        # Advanced insights
        normal_cols = [col for col, result in summary['normality_assessment'].items() 
                      if result.get('is_normal', False)]
        non_normal_cols = [col for col, result in summary['normality_assessment'].items() 
                          if not result.get('is_normal', False)]
        
        summary['advanced_insights'] = {
            'parametric_ready_columns': normal_cols,
            'non_parametric_columns': non_normal_cols,
            'potential_outlier_columns': [col for col, result in summary['normality_assessment'].items() 
                                        if result.get('outliers_percentage', 0) > 5],
            'skewed_columns': [col for col, result in summary['normality_assessment'].items() 
                             if abs(result.get('skewness', 0)) > 1]
        }
        
        # Generate recommendations
        recommendations = []
        
        if non_normal_cols:
            recommendations.append(
                f"Consider non-parametric tests for: {', '.join(non_normal_cols)}"
            )
        
        if high_missing_cols:
            recommendations.append(
                f"Address missing data in: {', '.join(high_missing_cols[:3])}"
            )
        
        if summary['dataset_overview']['duplicate_rows'] > 0:
            recommendations.append(
                f"Remove {summary['dataset_overview']['duplicate_rows']} duplicate rows"
            )
        
        if len(self.df) < 30:
            recommendations.append("Small sample size detected - consider non-parametric tests or collect more data")
        elif len(self.df) > 1000:
            recommendations.append("Large sample size - even small effects may be statistically significant")
        
        if summary['correlation_analysis'].get('strong_correlations_count', 0) > 0:
            recommendations.append("Strong correlations detected - consider multicollinearity in modeling")
        
        summary['recommendations'] = recommendations
        
        return summary