import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class BusinessKPIAnalyzer:
    """Enhanced Business KPI and insights generation utilities with advanced analytics"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Enhanced data type detection
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        
        # Data quality assessment
        self.data_quality_report = self._assess_data_quality()
        
        # Cache for expensive computations
        self._cache = {}
    
    def _assess_data_quality(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment with enhanced metrics"""
        report = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'completeness_score': 0,
            'quality_issues': [],
            'column_analysis': {}
        }
        
        # Calculate completeness score
        total_cells = len(self.df) * len(self.df.columns)
        if total_cells > 0:
            report['completeness_score'] = ((total_cells - report['missing_values']) / total_cells) * 100
        
        # Column-level analysis
        for col in self.df.columns:
            col_analysis = {
                'dtype': str(self.df[col].dtype),
                'missing_count': self.df[col].isnull().sum(),
                'missing_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique_count': self.df[col].nunique(),
                'constant_value': self.df[col].nunique() == 1
            }
            
            if self.df[col].dtype in [np.number, 'int64', 'float64']:
                col_analysis.update({
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'zeros_count': (self.df[col] == 0).sum(),
                    'negative_count': (self.df[col] < 0).sum()
                })
            
            report['column_analysis'][col] = col_analysis
            
            # Identify quality issues
            if col_analysis['missing_percentage'] > 20:
                report['quality_issues'].append(f"Column '{col}' has {col_analysis['missing_percentage']:.1f}% missing values")
            
            if col_analysis['constant_value']:
                report['quality_issues'].append(f"Column '{col}' has constant values")
        
        # Check for duplicate rows
        if report['duplicate_rows'] > 0:
            report['quality_issues'].append(f"{report['duplicate_rows']} duplicate rows found")
        
        return report

    def calculate_basic_kpis(self, value_col: str, quantity_col: Optional[str] = None,
                           date_col: Optional[str] = None) -> Dict[str, Any]:
        """Original method for backward compatibility"""
        return self.calculate_comprehensive_kpis(value_col, quantity_col, None, date_col, None)
    
    def calculate_comprehensive_kpis(self, value_col: str, quantity_col: Optional[str] = None,
                                   cost_col: Optional[str] = None, date_col: Optional[str] = None,
                                   customer_col: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive business KPIs with enhanced error handling"""
        kpis = {}
        
        # Basic validation
        if value_col not in self.df.columns:
            raise ValueError(f"Value column '{value_col}' not found in dataset")
        
        if value_col not in self.numeric_cols:
            # Try to convert to numeric
            try:
                self.df[value_col] = pd.to_numeric(self.df[value_col], errors='coerce')
                self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            except:
                raise ValueError(f"Value column '{value_col}' must be numeric or convertible to numeric")
        
        # Core value metrics with enhanced calculations
        valid_values = self.df[value_col].dropna()
        if len(valid_values) == 0:
            raise ValueError(f"No valid numeric values found in column '{value_col}'")
        
        kpis['total_revenue'] = valid_values.sum()
        kpis['average_value'] = valid_values.mean()
        kpis['median_value'] = valid_values.median()
        kpis['std_value'] = valid_values.std()
        kpis['min_value'] = valid_values.min()
        kpis['max_value'] = valid_values.max()
        kpis['transaction_count'] = len(valid_values)
        kpis['nonzero_count'] = (valid_values != 0).sum()
        
        # Enhanced variability metrics
        if kpis['average_value'] != 0:
            kpis['coefficient_of_variation'] = (kpis['std_value'] / kpis['average_value']) * 100
        else:
            kpis['coefficient_of_variation'] = 0
            
        kpis['gini_coefficient'] = self._calculate_gini(valid_values)
        
        # Quantity metrics with validation
        if quantity_col and quantity_col in self.df.columns:
            try:
                quantity_series = pd.to_numeric(self.df[quantity_col], errors='coerce').dropna()
                if len(quantity_series) > 0:
                    kpis['total_quantity'] = quantity_series.sum()
                    kpis['average_quantity'] = quantity_series.mean()
                    if kpis['total_quantity'] > 0:
                        kpis['avg_price'] = kpis['total_revenue'] / kpis['total_quantity']
                    else:
                        kpis['avg_price'] = 0
            except Exception as e:
                kpis['total_quantity'] = None
                kpis['avg_price'] = None
        
        # Financial metrics with enhanced calculations
        if cost_col and cost_col in self.df.columns:
            try:
                cost_series = pd.to_numeric(self.df[cost_col], errors='coerce').dropna()
                if len(cost_series) > 0:
                    kpis['total_cost'] = cost_series.sum()
                    kpis['total_profit'] = kpis['total_revenue'] - kpis['total_cost']
                    
                    if kpis['total_revenue'] > 0:
                        kpis['profit_margin'] = (kpis['total_profit'] / kpis['total_revenue']) * 100
                    else:
                        kpis['profit_margin'] = 0
                    
                    if kpis['total_cost'] > 0:
                        kpis['roi'] = (kpis['total_profit'] / kpis['total_cost']) * 100
                    else:
                        kpis['roi'] = 0
                    
                    if kpis.get('avg_price', 0) > 0:
                        kpis['breakeven_point'] = kpis['total_cost'] / kpis['avg_price']
                    else:
                        kpis['breakeven_point'] = 0
            except Exception as e:
                # Silently skip cost metrics if calculation fails
                pass
        
        # Customer metrics with validation
        if customer_col and customer_col in self.df.columns:
            customer_data = self.df[customer_col].dropna()
            if len(customer_data) > 0:
                kpis['unique_customers'] = customer_data.nunique()
                if kpis['unique_customers'] > 0:
                    kpis['avg_customer_value'] = kpis['total_revenue'] / kpis['unique_customers']
                    kpis['purchase_frequency'] = kpis['transaction_count'] / kpis['unique_customers']
                    kpis['estimated_clv'] = kpis['avg_customer_value'] * kpis['purchase_frequency'] * 12  # Annual estimate
        
        # Time-based growth metrics with enhanced period handling
        if date_col and date_col in self.df.columns:
            try:
                # Ensure date column is properly formatted
                date_series = pd.to_datetime(self.df[date_col], errors='coerce')
                valid_dates = date_series.dropna()
                if len(valid_dates) > 0:
                    temp_df = self.df.copy()
                    temp_df[date_col] = date_series
                    growth_metrics = self._calculate_comprehensive_growth(temp_df, date_col, value_col, quantity_col)
                    kpis.update(growth_metrics)
            except Exception as e:
                # Silently skip growth metrics if calculation fails
                pass
        
        # Enhanced concentration metrics
        if len(valid_values) > 0:
            top_10_count = max(1, int(len(valid_values) * 0.1))
            top_10_sum = valid_values.nlargest(top_10_count).sum()
            kpis['top_10_concentration'] = (top_10_sum / kpis['total_revenue']) * 100 if kpis['total_revenue'] > 0 else 0
            
            # Skewness and kurtosis for distribution insights
            kpis['skewness'] = valid_values.skew()
            kpis['kurtosis'] = valid_values.kurtosis()
        
        # Data quality metrics
        kpis['data_quality_score'] = self.data_quality_report['completeness_score']
        kpis['data_quality_issues'] = len(self.data_quality_report['quality_issues'])
        
        return kpis
    
    def _calculate_comprehensive_growth(self, df: pd.DataFrame, date_col: str, value_col: str, 
                                      quantity_col: Optional[str] = None) -> Dict[str, float]:
        """Calculate comprehensive growth metrics with enhanced period handling"""
        growth_metrics = {}
        
        try:
            # Ensure date is sorted and timezone-naive
            df_sorted = df.sort_values(date_col).copy()
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col]).dt.tz_localize(None)
            
            # Remove rows with invalid dates or values
            df_sorted = df_sorted[df_sorted[date_col].notna() & df_sorted[value_col].notna()]
            
            if len(df_sorted) < 2:
                return growth_metrics
            
            # Current vs previous period comparisons
            current_date = df_sorted[date_col].max()
            
            # Monthly growth calculation
            current_month = current_date.replace(day=1)
            prev_month = (current_month - pd.DateOffset(months=1)).replace(day=1)
            
            current_month_data = df_sorted[df_sorted[date_col] >= current_month]
            prev_month_data = df_sorted[
                (df_sorted[date_col] >= prev_month) & 
                (df_sorted[date_col] < current_month)
            ]
            
            if len(prev_month_data) > 0 and len(current_month_data) > 0:
                # Revenue growth
                current_revenue = current_month_data[value_col].sum()
                prev_revenue = prev_month_data[value_col].sum()
                if prev_revenue > 0:
                    growth_metrics['revenue_growth'] = ((current_revenue - prev_revenue) / prev_revenue) * 100
                
                # Value growth
                current_avg = current_month_data[value_col].mean()
                prev_avg = prev_month_data[value_col].mean()
                if prev_avg > 0:
                    growth_metrics['avg_value_growth'] = ((current_avg - prev_avg) / prev_avg) * 100
                
                # Quantity growth
                if quantity_col and quantity_col in df_sorted.columns:
                    current_qty = current_month_data[quantity_col].sum()
                    prev_qty = prev_month_data[quantity_col].sum()
                    if prev_qty > 0:
                        growth_metrics['quantity_growth'] = ((current_qty - prev_qty) / prev_qty) * 100
                
                # Transaction growth
                current_count = len(current_month_data)
                prev_count = len(prev_month_data)
                if prev_count > 0:
                    growth_metrics['transaction_growth'] = ((current_count - prev_count) / prev_count) * 100
            
            # YTD growth calculation
            ytd_start = current_date.replace(month=1, day=1)
            ytd_data = df_sorted[df_sorted[date_col] >= ytd_start]
            
            if current_date.year > df_sorted[date_col].min().year:
                prev_ytd_start = ytd_start.replace(year=ytd_start.year-1)
                prev_ytd_end = current_date.replace(year=current_date.year-1)
                prev_ytd_data = df_sorted[
                    (df_sorted[date_col] >= prev_ytd_start) &
                    (df_sorted[date_col] <= prev_ytd_end)
                ]
                
                if len(prev_ytd_data) > 0 and prev_ytd_data[value_col].sum() > 0:
                    ytd_growth = ((ytd_data[value_col].sum() - prev_ytd_data[value_col].sum()) / 
                                 prev_ytd_data[value_col].sum()) * 100
                    growth_metrics['ytd_growth'] = ytd_growth
        
        except Exception as e:
            # Silently handle errors in growth calculation
            pass
        
        return growth_metrics
    
    def _calculate_gini(self, series: pd.Series) -> float:
        """Calculate Gini coefficient for inequality measurement with enhanced robustness"""
        try:
            # Sort the series and remove NaN, zero, and negative values for meaningful Gini
            sorted_series = np.sort(series[series > 0].dropna())
            n = len(sorted_series)
            if n == 0:
                return 0
            
            index = np.arange(1, n + 1)
            gini = (np.sum((2 * index - n - 1) * sorted_series)) / (n * np.sum(sorted_series))
            return min(max(gini, 0), 1)  # Ensure between 0 and 1
        except:
            return 0
    
    def analyze_trends(self, value_col: str, date_col: str, period: str = 'M') -> pd.Series:
        """Analyze trends over time with enhanced period handling and error recovery"""
        if date_col not in self.df.columns or value_col not in self.df.columns:
            return None
        
        try:
            # Create working copy and ensure proper data types
            temp_df = self.df[[date_col, value_col]].copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
            temp_df[value_col] = pd.to_numeric(temp_df[value_col], errors='coerce')
            
            # Remove invalid rows
            temp_df = temp_df[temp_df[date_col].notna() & temp_df[value_col].notna()]
            
            if len(temp_df) == 0:
                return None
            
            trend_data = temp_df.set_index(date_col).sort_index()
            
            period_map = {
                'D': 'D',
                'Weekly': 'W-MON',
                'Monthly': 'M',
                'Quarterly': 'Q',
                'Yearly': 'Y'
            }
            
            resample_period = period_map.get(period, 'M')
            
            # Enhanced resampling with appropriate aggregation
            try:
                if period in ['D', 'Weekly']:
                    resampled = trend_data[value_col].resample(resample_period).mean()
                else:
                    resampled = trend_data[value_col].resample(resample_period).mean()
                
                # Remove any NaN values that might have been introduced
                return resampled.dropna()
                
            except Exception as e:
                # Fallback to simple grouping if resampling fails
                if period == 'Monthly':
                    trend_data['year_month'] = trend_data.index.strftime('%Y-%m')
                    return trend_data.groupby('year_month')[value_col].mean()
                else:
                    return trend_data[value_col]
        
        except Exception as e:
            return None
    
    def generate_forecast(self, trend_data: pd.Series, periods: int = 3) -> Optional[pd.Series]:
        """Generate enhanced forecast using multiple methods with fallbacks"""
        if len(trend_data) < 3:
            return None
        
        try:
            # Prepare data for regression
            X = np.arange(len(trend_data)).reshape(-1, 1)
            y = trend_data.values
            
            # Handle NaN/inf values
            valid_mask = ~np.isnan(y) & ~np.isinf(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 3:
                return None
            
            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate forecast
            future_X = np.arange(len(trend_data), len(trend_data) + periods).reshape(-1, 1)
            future_y = model.predict(future_X)
            
            # Ensure forecast values are reasonable
            future_y = np.maximum(future_y, 0)  # No negative forecasts for business metrics
            
            # Create forecast index
            last_date = trend_data.index[-1]
            
            # Determine frequency for forecast index
            if len(trend_data) > 1:
                time_diffs = np.diff(trend_data.index)
                if len(time_diffs) > 0:
                    avg_freq = np.mean([td.total_seconds() for td in time_diffs])
                    if avg_freq >= 2592000:  # ~30 days
                        freq = 'M'
                    elif avg_freq >= 604800:  # 7 days
                        freq = 'W'
                    else:
                        freq = 'D'
                else:
                    freq = 'M'
            else:
                freq = 'M'
            
            future_dates = pd.date_range(last_date, periods=periods+1, freq=freq)[1:]
            
            return pd.Series(future_y, index=future_dates)
        
        except Exception as e:
            return None
    
    def calculate_growth_rates(self, trend_data: pd.Series, value_col: str) -> Dict[str, float]:
        """Original method for backward compatibility"""
        return self.calculate_comprehensive_growth_rates(trend_data, value_col)
    
    def calculate_comprehensive_growth_rates(self, trend_data: pd.Series, value_col: str) -> Dict[str, float]:
        """Calculate comprehensive growth rates with enhanced confidence intervals"""
        growth_rates = {}
        
        if len(trend_data) < 2:
            return growth_rates
        
        try:
            # Ensure we have valid numeric data
            valid_data = trend_data.dropna()
            if len(valid_data) < 2:
                return growth_rates
            
            # MTD growth (if we have sufficient data)
            if len(valid_data) >= 10:
                mtd_data = valid_data.iloc[-10:]  # Last 10 periods
                if len(mtd_data) > 1:
                    mtd_growth = ((mtd_data.iloc[-1] - mtd_data.iloc[0]) / abs(mtd_data.iloc[0])) * 100
                    growth_rates['mtd_growth'] = mtd_growth
            
            # Monthly growth (last two periods)
            if len(valid_data) >= 2:
                monthly_growth = ((valid_data.iloc[-1] - valid_data.iloc[-2]) / abs(valid_data.iloc[-2])) * 100
                growth_rates['monthly_growth'] = monthly_growth
            
            # Quarterly growth (3-period average)
            if len(valid_data) >= 6:
                quarterly_avg_current = valid_data.iloc[-3:].mean()
                quarterly_avg_previous = valid_data.iloc[-6:-3].mean()
                if abs(quarterly_avg_previous) > 0:
                    quarterly_growth = ((quarterly_avg_current - quarterly_avg_previous) / abs(quarterly_avg_previous)) * 100
                    growth_rates['quarterly_growth'] = quarterly_growth
            
            # YTD growth (year-to-date approximation)
            if len(valid_data) >= 6:
                ytd_avg = valid_data.iloc[-6:].mean()
                beginning_avg = valid_data.iloc[:6].mean() if len(valid_data) >= 12 else valid_data.iloc[0]
                if abs(beginning_avg) > 0:
                    ytd_growth = ((ytd_avg - beginning_avg) / abs(beginning_avg)) * 100
                    growth_rates['ytd_growth'] = ytd_growth
            
            # Yearly growth (if sufficient data)
            if len(valid_data) >= 13:
                yearly_avg_current = valid_data.iloc[-12:].mean()
                yearly_avg_previous = valid_data.iloc[-24:-12].mean() if len(valid_data) >= 24 else valid_data.iloc[0]
                if abs(yearly_avg_previous) > 0:
                    yearly_growth = ((yearly_avg_current - yearly_avg_previous) / abs(yearly_avg_previous)) * 100
                    growth_rates['yearly_growth'] = yearly_growth
            
            # Overall trend growth
            if len(valid_data) >= 2:
                overall_growth = ((valid_data.iloc[-1] - valid_data.iloc[0]) / abs(valid_data.iloc[0])) * 100
                growth_rates['overall_growth'] = overall_growth
        
        except Exception as e:
            # Silently handle calculation errors
            pass
        
        return growth_rates
    
    def analyze_seasonality(self, trend_data: pd.Series, value_col: str) -> Optional[pd.DataFrame]:
        """Enhanced seasonality analysis with statistical significance and robustness"""
        if len(trend_data) < 12:
            return None
        
        try:
            seasonal_data = trend_data.copy().to_frame()
            seasonal_data.columns = [value_col]
            
            # Extract time components
            seasonal_data['month'] = seasonal_data.index.month
            seasonal_data['year'] = seasonal_data.index.year
            
            # Remove any incomplete years
            year_counts = seasonal_data['year'].value_counts()
            complete_years = year_counts[year_counts >= 12].index
            seasonal_data = seasonal_data[seasonal_data['year'].isin(complete_years)]
            
            if len(seasonal_data) == 0:
                return None
            
            # Calculate monthly statistics
            monthly_stats = seasonal_data.groupby('month')[value_col].agg(['mean', 'std', 'count']).dropna()
            
            if len(monthly_stats) < 6:  # Need at least 6 months for meaningful analysis
                return None
            
            # Calculate standard error and confidence intervals
            monthly_stats['se'] = monthly_stats['std'] / np.sqrt(monthly_stats['count'])
            monthly_stats['ci_lower'] = monthly_stats['mean'] - 1.96 * monthly_stats['se']
            monthly_stats['ci_upper'] = monthly_stats['mean'] + 1.96 * monthly_stats['se']
            
            return monthly_stats['mean']
        
        except Exception as e:
            return None
    
    def generate_seasonal_insights(self, seasonality_data: pd.Series) -> List[str]:
        """Generate enhanced insights from seasonality patterns"""
        insights = []
        
        if seasonality_data is None or seasonality_data.empty:
            insights.append("No significant seasonal patterns detected in the available data.")
            return insights
        
        try:
            if len(seasonality_data) < 6:
                insights.append("Insufficient data for robust seasonal analysis.")
                return insights
            
            peak_month = seasonality_data.idxmax()
            trough_month = seasonality_data.idxmin()
            seasonality_strength = ((seasonality_data.max() - seasonality_data.min()) / seasonality_data.mean()) * 100
            
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            
            peak_month_name = month_names[peak_month - 1] if 1 <= peak_month <= 12 else f"Month {peak_month}"
            trough_month_name = month_names[trough_month - 1] if 1 <= trough_month <= 12 else f"Month {trough_month}"
            
            peak_performance = ((seasonality_data.max() - seasonality_data.mean()) / seasonality_data.mean()) * 100
            trough_performance = ((seasonality_data.min() - seasonality_data.mean()) / seasonality_data.mean()) * 100
            
            insights.append(f"📈 Peak performance in {peak_month_name} ({peak_performance:+.1f}% above average)")
            insights.append(f"📉 Lowest performance in {trough_month_name} ({trough_performance:+.1f}% below average)")
            
            if seasonality_strength > 50:
                insights.append("🎯 Strong seasonality detected - consider seasonal planning and resource allocation strategies")
            elif seasonality_strength > 20:
                insights.append("📊 Moderate seasonality present - account for monthly variations in planning")
            else:
                insights.append("⚖️ Minimal seasonality - relatively consistent performance throughout the year")
            
            # Additional insight: identify consecutive strong/weak months
            above_avg = seasonality_data > seasonality_data.mean()
            if above_avg.any():
                consecutive_strong = self._find_consecutive_months(above_avg)
                if consecutive_strong:
                    insights.append(f"🔍 Strong consecutive months: {', '.join(consecutive_strong)}")
            
        except Exception as e:
            insights.append("Seasonal analysis completed with limited insights due to data constraints.")
        
        return insights
    
    def _find_consecutive_months(self, month_mask: pd.Series) -> List[str]:
        """Find consecutive months in a boolean series"""
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        groups = []
        current_group = []
        
        for i, (month_idx, is_strong) in enumerate(month_mask.items()):
            if is_strong:
                current_group.append(month_names[month_idx - 1])
            elif current_group:
                if len(current_group) >= 2:  # Only consider groups of 2 or more
                    groups.append(current_group)
                current_group = []
        
        if current_group and len(current_group) >= 2:
            groups.append(current_group)
        
        return ['-'.join(group) for group in groups]
    
    def calculate_trend_statistics(self, trend_data: pd.Series) -> Dict[str, Any]:
        """Calculate comprehensive trend statistics with enhanced metrics"""
        stats = {}
        
        if len(trend_data) < 3:
            return stats
        
        try:
            valid_data = trend_data.dropna()
            if len(valid_data) < 3:
                return stats
            
            # Trend strength (R-squared of linear trend)
            X = np.arange(len(valid_data)).reshape(-1, 1)
            y = valid_data.values
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            stats['trend_strength'] = r_squared
            
            # Volatility (coefficient of variation)
            if valid_data.mean() != 0:
                stats['volatility'] = (valid_data.std() / valid_data.mean()) * 100
            else:
                stats['volatility'] = 0
            
            # Best and worst periods
            best_period = valid_data.idxmax()
            worst_period = valid_data.idxmin()
            stats['best_period'] = best_period.strftime('%B %Y')
            stats['worst_period'] = worst_period.strftime('%B %Y')
            stats['best_value'] = valid_data.max()
            stats['worst_value'] = valid_data.min()
            
            # Trend direction and magnitude
            trend_slope = model.coef_[0]
            stats['trend_direction'] = 'Upward' if trend_slope > 0 else 'Downward'
            stats['trend_magnitude'] = abs(trend_slope)
            stats['trend_slope'] = trend_slope
            
            # Additional statistics
            stats['mean'] = valid_data.mean()
            stats['std_dev'] = valid_data.std()
            stats['data_points'] = len(valid_data)
            
            # Trend consistency (percentage of periods with growth)
            if len(valid_data) > 1:
                growth_periods = (np.diff(valid_data.values) > 0).sum()
                stats['growth_consistency'] = (growth_periods / (len(valid_data) - 1)) * 100
        
        except Exception as e:
            # Silently handle calculation errors
            pass
        
        return stats
    
    def analyze_segment_performance(self, metric_col: str, segment_col: str) -> pd.DataFrame:
        """Original method for backward compatibility"""
        return self.analyze_segment_performance_with_stats(metric_col, segment_col)
    
    def analyze_segment_performance_with_stats(self, metric_col: str, segment_col: str) -> pd.DataFrame:
        """Enhanced segment performance analysis with comprehensive statistical significance"""
        # Validate inputs
        if metric_col not in self.df.columns or segment_col not in self.df.columns:
            return pd.DataFrame()
        
        try:
            # Clean data
            clean_df = self.df[[segment_col, metric_col]].dropna()
            if len(clean_df) == 0:
                return pd.DataFrame()
            
            # Convert metric to numeric if needed
            clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors='coerce')
            clean_df = clean_df.dropna()
            
            if len(clean_df) == 0:
                return pd.DataFrame()
            
            # Calculate segment statistics
            segment_stats = clean_df.groupby(segment_col)[metric_col].agg([
                'mean', 'sum', 'count', 'std', 'min', 'max'
            ]).sort_values('mean', ascending=False)
            
            # Calculate additional metrics
            overall_mean = clean_df[metric_col].mean()
            segment_stats['percentage_of_total'] = (segment_stats['sum'] / segment_stats['sum'].sum()) * 100
            
            # Statistical significance testing
            segment_stats['z_score'] = (segment_stats['mean'] - overall_mean) / (
                segment_stats['std'] / np.sqrt(segment_stats['count'])
            )
            segment_stats['p_value'] = segment_stats['z_score'].apply(
                lambda z: 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else 1.0
            )
            segment_stats['significance'] = segment_stats['p_value'].apply(
                lambda p: 'High' if p < 0.01 else 'Medium' if p < 0.05 else 'Low'
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((segment_stats['count'] - 1) * segment_stats['std']**2).sum() / 
                               (segment_stats['count'].sum() - len(segment_stats)))
            if pooled_std > 0:
                segment_stats['effect_size'] = (segment_stats['mean'] - overall_mean) / pooled_std
            else:
                segment_stats['effect_size'] = 0
            
            return segment_stats.round(4)
        
        except Exception as e:
            return pd.DataFrame()
    
    def pareto_analysis(self, value_col: str, segment_col: str) -> Optional[pd.DataFrame]:
        """Perform enhanced Pareto analysis (80/20 rule) with validation"""
        if value_col not in self.df.columns or segment_col not in self.df.columns:
            return None
        
        try:
            # Clean and prepare data
            clean_df = self.df[[segment_col, value_col]].dropna()
            clean_df[value_col] = pd.to_numeric(clean_df[value_col], errors='coerce')
            clean_df = clean_df.dropna()
            
            if len(clean_df) == 0:
                return None
            
            segment_performance = clean_df.groupby(segment_col)[value_col].sum().sort_values(ascending=False)
            
            if len(segment_performance) == 0:
                return None
            
            # Calculate cumulative percentages
            total_value = segment_performance.sum()
            cumulative_value = segment_performance.cumsum()
            cumulative_percentage = (cumulative_value / total_value) * 100
            
            pareto_data = pd.DataFrame({
                'segment': segment_performance.index,
                'value': segment_performance.values,
                'cumulative_value': cumulative_value.values,
                'cumulative_percentage': cumulative_percentage.values,
                'percentage_of_total': (segment_performance.values / total_value) * 100
            })
            
            return pareto_data.reset_index(drop=True)
        
        except Exception as e:
            return None
    
    def performance_tiers(self, performance_col: str, tiers: int = 4) -> Dict[str, int]:
        """Enhanced performance tier categorization with robust percentile calculation"""
        if performance_col not in self.df.columns:
            return {}
        
        try:
            # Clean data
            clean_data = self.df[performance_col].dropna()
            if len(clean_data) == 0:
                return {}
            
            # Convert to numeric
            clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
            if len(clean_data) == 0:
                return {}
            
            # Validate tiers parameter
            if tiers not in [3, 4, 5]:
                tiers = 4
            
            # Calculate percentiles
            percentiles = np.linspace(0, 100, tiers + 1)
            thresholds = np.percentile(clean_data, percentiles)
            
            # Define tier names
            tier_names = {
                3: ['Bottom', 'Middle', 'Top'],
                4: ['Bottom', 'Low', 'High', 'Top'],
                5: ['Bottom', 'Low', 'Middle', 'High', 'Top']
            }[tiers]
            
            tier_counts = {}
            
            # Count records in each tier
            for i in range(tiers):
                if i == 0:
                    mask = clean_data <= thresholds[i+1]
                elif i == tiers - 1:
                    mask = clean_data > thresholds[i]
                else:
                    mask = (clean_data > thresholds[i]) & (clean_data <= thresholds[i+1])
                
                tier_counts[tier_names[i]] = mask.sum()
            
            return tier_counts
        
        except Exception as e:
            return {}
    
    def benchmark_analysis(self, metric_col: str, benchmark_col: str) -> Optional[pd.DataFrame]:
        """Original method for backward compatibility"""
        return self.benchmark_analysis_with_stats(metric_col, benchmark_col)
    
    def benchmark_analysis_with_stats(self, metric_col: str, benchmark_col: str) -> pd.DataFrame:
        """Enhanced benchmark analysis with comprehensive statistical testing"""
        if metric_col not in self.df.columns or benchmark_col not in self.df.columns:
            return pd.DataFrame()
        
        try:
            # Clean data
            clean_df = self.df[[benchmark_col, metric_col]].dropna()
            clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors='coerce')
            clean_df = clean_df.dropna()
            
            if len(clean_df) == 0:
                return pd.DataFrame()
            
            # Calculate benchmark statistics
            benchmark_stats = clean_df.groupby(benchmark_col)[metric_col].agg([
                'mean', 'median', 'std', 'count', 'min', 'max'
            ])
            
            # Additional metrics
            benchmark_stats['coefficient_of_variation'] = (benchmark_stats['std'] / benchmark_stats['mean']) * 100
            benchmark_stats['percentage_of_total'] = (benchmark_stats['mean'] * benchmark_stats['count']) / (
                benchmark_stats['mean'] * benchmark_stats['count']).sum() * 100
            
            # ANOVA test for group differences
            groups = [
                group[metric_col].values 
                for name, group in clean_df.groupby(benchmark_col) 
                if len(group) > 1
            ]
            
            if len(groups) >= 2:
                try:
                    f_stat, p_value = stats.f_oneway(*groups)
                    benchmark_stats.attrs['anova_f_stat'] = f_stat
                    benchmark_stats.attrs['anova_p_value'] = p_value
                    benchmark_stats.attrs['groups_significant'] = p_value < 0.05
                except:
                    benchmark_stats.attrs['anova_f_stat'] = None
                    benchmark_stats.attrs['anova_p_value'] = None
                    benchmark_stats.attrs['groups_significant'] = False
            
            return benchmark_stats.sort_values('mean', ascending=False).round(4)
        
        except Exception as e:
            return pd.DataFrame()
    
    def calculate_performance_gaps(self, metric_col: str, segment_col: str) -> List[Dict[str, Any]]:
        """Calculate comprehensive performance gaps between segments"""
        if metric_col not in self.df.columns or segment_col not in self.df.columns:
            return []
        
        try:
            # Clean data
            clean_df = self.df[[segment_col, metric_col]].dropna()
            clean_df[metric_col] = pd.to_numeric(clean_df[metric_col], errors='coerce')
            clean_df = clean_df.dropna()
            
            if len(clean_df) == 0:
                return []
            
            segment_means = clean_df.groupby(segment_col)[metric_col].mean()
            overall_mean = clean_df[metric_col].mean()
            
            gaps = []
            for segment, mean_value in segment_means.items():
                if overall_mean != 0:
                    gap_percentage = ((mean_value - overall_mean) / overall_mean) * 100
                else:
                    gap_percentage = 0
                
                gaps.append({
                    'segment': segment,
                    'segment_mean': mean_value,
                    'overall_mean': overall_mean,
                    'gap_value': mean_value - overall_mean,
                    'gap_percentage': gap_percentage,
                    'performance': 'above' if gap_percentage > 0 else 'below',
                    'significance': 'high' if abs(gap_percentage) > 20 else 'medium' if abs(gap_percentage) > 10 else 'low'
                })
            
            return sorted(gaps, key=lambda x: abs(x['gap_percentage']), reverse=True)
        
        except Exception as e:
            return []
    
    def driver_analysis(self, outcome_col: str, driver_cols: List[str]) -> Dict[str, float]:
        """Original method for backward compatibility"""
        comprehensive = self.comprehensive_driver_analysis(outcome_col, driver_cols)
        return comprehensive.get('correlations', {})
    
    def comprehensive_driver_analysis(self, outcome_col: str, driver_cols: List[str]) -> Dict[str, Any]:
        """Comprehensive driver analysis with enhanced statistical significance and validation"""
        analysis = {
            'correlations': {},
            'significant_drivers': [],
            'top_drivers': [],
            'analysis_quality': 'Good'
        }
        
        if outcome_col not in self.df.columns:
            analysis['analysis_quality'] = 'Poor - outcome column missing'
            return analysis
        
        valid_drivers = [driver for driver in driver_cols if driver in self.df.columns and driver != outcome_col]
        
        if len(valid_drivers) == 0:
            analysis['analysis_quality'] = 'Poor - no valid driver columns'
            return analysis
        
        try:
            # Prepare data for analysis
            analysis_cols = valid_drivers + [outcome_col]
            clean_df = self.df[analysis_cols].dropna()
            
            # Convert to numeric
            for col in analysis_cols:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
            clean_df = clean_df.dropna()
            
            if len(clean_df) < 10:
                analysis['analysis_quality'] = 'Poor - insufficient data after cleaning'
                return analysis
            
            # Calculate correlations with significance
            for driver in valid_drivers:
                try:
                    correlation, p_value = stats.pearsonr(clean_df[driver], clean_df[outcome_col])
                    
                    if not np.isnan(correlation) and not np.isnan(p_value):
                        analysis['correlations'][driver] = correlation
                        
                        if p_value < 0.05:
                            strength = 'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'
                            
                            analysis['significant_drivers'].append({
                                'driver': driver,
                                'correlation': correlation,
                                'p_value': p_value,
                                'strength': strength,
                                'direction': 'positive' if correlation > 0 else 'negative'
                            })
                except:
                    continue
            
            # Identify top drivers
            analysis['top_drivers'] = sorted(
                analysis['significant_drivers'],
                key=lambda x: abs(x['correlation']),
                reverse=True
            )[:5]
            
            # Calculate analysis quality metrics
            valid_correlations = len([c for c in analysis['correlations'].values() if not np.isnan(c)])
            analysis['analysis_quality'] = 'Excellent' if valid_correlations >= 5 else 'Good' if valid_correlations >= 3 else 'Fair'
            
            # Additional insights
            if analysis['significant_drivers']:
                strongest_driver = analysis['top_drivers'][0] if analysis['top_drivers'] else None
                if strongest_driver:
                    analysis['primary_driver'] = strongest_driver['driver']
                    analysis['primary_relationship'] = strongest_driver['direction']
            
            analysis['data_points_used'] = len(clean_df)
            analysis['drivers_analyzed'] = len(valid_drivers)
            analysis['significant_found'] = len(analysis['significant_drivers'])
            
        except Exception as e:
            analysis['analysis_quality'] = f'Error: {str(e)}'
        
        return analysis
    
    def generate_strategic_insights(self) -> Dict[str, Any]:
        """Original method for backward compatibility"""
        return self.generate_comprehensive_strategic_insights()
    
    def generate_comprehensive_strategic_insights(self, depth: str = "Standard Analysis", 
                                                focus_area: str = "All Areas",
                                                time_horizon: str = "Medium-term (3-12 months)") -> Dict[str, Any]:
        """Generate comprehensive strategic insights with AI-like analysis and enhanced business context"""
        insights = {
            'performance_insights': [],
            'growth_opportunities': [],
            'risk_alerts': [],
            'efficiency_insights': [],
            'customer_insights': [],
            'competitive_insights': [],
            'data_quality_insights': [],
            'analysis_context': {
                'depth': depth,
                'focus_area': focus_area,
                'time_horizon': time_horizon,
                'data_points': len(self.df),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Basic data assessment
        total_rows = len(self.df)
        if total_rows == 0:
            insights['performance_insights'].append("No data available for analysis")
            return insights
        
        # Data Quality Insights
        data_quality_score = self.data_quality_report['completeness_score']
        if data_quality_score < 80:
            insights['data_quality_insights'].append(
                f"Data quality score is {data_quality_score:.1f}% - consider improving data collection processes"
            )
        
        if self.data_quality_report['missing_values'] > 0:
            insights['data_quality_insights'].append(
                f"Dataset contains {self.data_quality_report['missing_values']} missing values - may affect analysis accuracy"
            )
        
        # Performance insights based on numeric columns
        if self.numeric_cols:
            primary_metric = self.numeric_cols[0]
            metric_stats = self.df[primary_metric].describe()
            
            # Performance distribution insights
            skewness = self.df[primary_metric].skew()
            if abs(skewness) > 1:
                direction = "right" if skewness > 0 else "left"
                insights['performance_insights'].append(
                    f"Data shows strong {direction}-skewed distribution (skewness: {skewness:.2f}), "
                    f"suggesting a few high performers drive most results. Consider segment-specific strategies."
                )
            elif abs(skewness) > 0.5:
                insights['performance_insights'].append(
                    f"Moderate distribution skewness ({skewness:.2f}) indicates some performance concentration."
                )
            
            # Value concentration insights
            top_10_count = max(1, int(len(self.df) * 0.1))
            top_10_pct = self.df[primary_metric].nlargest(top_10_count).sum()
            total_value = self.df[primary_metric].sum()
            concentration_ratio = (top_10_pct / total_value) * 100 if total_value > 0 else 0
            
            if concentration_ratio > 80:
                insights['performance_insights'].append(
                    f"High concentration detected: Top 10% of records contribute {concentration_ratio:.1f}% of total value - "
                    f"consider diversifying revenue sources and developing middle performers"
                )
            elif concentration_ratio > 60:
                insights['performance_insights'].append(
                    f"Moderate concentration: Top 10% contribute {concentration_ratio:.1f}% of value - "
                    f"healthy balance between top performers and broader base"
                )
        
        # Growth insights with time series analysis
        if self.date_cols and self.numeric_cols:
            date_col = self.date_cols[0]
            value_col = self.numeric_cols[0]
            trend_data = self.analyze_trends(value_col, date_col)
            
            if trend_data is not None and len(trend_data) > 1:
                overall_growth = ((trend_data.iloc[-1] - trend_data.iloc[0]) / abs(trend_data.iloc[0])) * 100
                
                if overall_growth > 20:
                    insights['growth_opportunities'].append(
                        f"Strong overall growth trend: {overall_growth:.1f}% increase over period - "
                        f"capitalize on growth momentum and consider scaling successful initiatives"
                    )
                elif overall_growth > 0:
                    insights['growth_opportunities'].append(
                        f"Positive growth trend: {overall_growth:.1f}% increase - "
                        f"maintain current strategies while exploring new opportunities"
                    )
                elif overall_growth < -10:
                    insights['risk_alerts'].append(
                        f"Declining trend: {abs(overall_growth):.1f}% decrease over period - "
                        f"investigate causes and implement recovery plan. Consider market changes and competitive pressures."
                    )
                
                # Recent trend analysis
                if len(trend_data) >= 3:
                    recent_growth = ((trend_data.iloc[-1] - trend_data.iloc[-2]) / abs(trend_data.iloc[-2])) * 100
                    if recent_growth > 10:
                        insights['growth_opportunities'].append(
                            f"Strong recent momentum: {recent_growth:.1f}% growth in latest period - "
                            f"leverage this acceleration for strategic advantage"
                        )
        
        # Efficiency insights from correlations
        if len(self.numeric_cols) >= 2:
            efficiency_pairs = []
            for i, col1 in enumerate(self.numeric_cols[:5]):  # Limit to first 5 for performance
                for col2 in self.numeric_cols[i+1:6]:  # Limit pairs for efficiency
                    try:
                        correlation = self.df[col1].corr(self.df[col2])
                        if abs(correlation) > 0.7:
                            efficiency_pairs.append((col1, col2, correlation))
                    except:
                        continue
            
            if efficiency_pairs:
                insights['efficiency_insights'].append(
                    f"Found {len(efficiency_pairs)} strongly correlated variable pairs - "
                    f"opportunity for process optimization and resource allocation. Analyze these relationships for efficiency gains."
                )
        
        # Customer insights (if customer-like column exists)
        customer_like_cols = [
            col for col in self.categorical_cols 
            if any(term in col.lower() for term in ['customer', 'client', 'user', 'id', 'account'])
        ]
        if customer_like_cols and self.numeric_cols:
            customer_col = customer_like_cols[0]
            value_col = self.numeric_cols[0]
            
            try:
                customer_stats = self.df.groupby(customer_col)[value_col].agg(['sum', 'count']).sort_values('sum', ascending=False)
                top_customers = customer_stats.head(5)
                
                if len(customer_stats) > 0 and customer_stats['sum'].sum() > 0:
                    concentration = top_customers['sum'].sum() / customer_stats['sum'].sum() * 100
                    if concentration > 50:
                        insights['customer_insights'].append(
                            f"Customer concentration: Top 5 customers represent {concentration:.1f}% of total value - "
                            f"consider customer diversification strategy and relationship strengthening with key accounts"
                        )
                    elif concentration < 20:
                        insights['customer_insights'].append(
                            f"Healthy customer distribution: Top 5 customers represent {concentration:.1f}% of value - "
                            f"well-diversified customer base reduces business risk"
                        )
            except:
                pass
        
        # Competitive insights based on segment performance
        if self.categorical_cols and self.numeric_cols:
            segment_col = self.categorical_cols[0]
            value_col = self.numeric_cols[0]
            
            try:
                segment_performance = self.analyze_segment_performance_with_stats(value_col, segment_col)
                if not segment_performance.empty and len(segment_performance) > 1:
                    best_segment = segment_performance.index[0]
                    worst_segment = segment_performance.index[-1]
                    performance_gap = (segment_performance.iloc[0]['mean'] - segment_performance.iloc[-1]['mean']) / segment_performance.iloc[-1]['mean'] * 100
                    
                    if performance_gap > 100:
                        insights['competitive_insights'].append(
                            f"Significant performance gap: {best_segment} outperforms {worst_segment} by {performance_gap:.1f}% - "
                            f"opportunity to replicate best practices across segments and address underperformance"
                        )
                    elif performance_gap > 50:
                        insights['competitive_insights'].append(
                            f"Moderate performance variation: {performance_gap:.1f}% gap between best and worst segments - "
                            f"standardize successful approaches where applicable"
                        )
            except:
                pass
        
        # Risk insights based on data patterns
        outlier_insights = self._generate_outlier_insights()
        insights['risk_alerts'].extend(outlier_insights)
        
        # Filter insights based on focus area if not "All Areas"
        if focus_area != "All Areas":
            filtered_insights = {}
            area_mapping = {
                'Growth': ['growth_opportunities', 'competitive_insights'],
                'Efficiency': ['efficiency_insights', 'performance_insights'],
                'Risk': ['risk_alerts', 'data_quality_insights'],
                'Customer Experience': ['customer_insights', 'performance_insights']
            }
            
            relevant_categories = area_mapping.get(focus_area, [])
            for category in insights.keys():
                if category in relevant_categories or category == 'analysis_context':
                    filtered_insights[category] = insights[category]
            
            return filtered_insights
        
        return insights
    
    def _generate_outlier_insights(self) -> List[str]:
        """Generate insights based on outlier detection"""
        outlier_insights = []
        
        if not self.numeric_cols:
            return outlier_insights
        
        for col in self.numeric_cols[:3]:  # Analyze first 3 numeric columns
            try:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                    outlier_pct = (outliers / len(self.df)) * 100
                    
                    if outlier_pct > 10:
                        outlier_insights.append(
                            f"High outlier prevalence in {col}: {outlier_pct:.1f}% of values are outliers - "
                            f"investigate data quality and business context"
                        )
                    elif outlier_pct > 5:
                        outlier_insights.append(
                            f"Moderate outliers in {col}: {outlier_pct:.1f}% of values - "
                            f"review for potential data issues or exceptional cases"
                        )
            except:
                continue
        
        return outlier_insights
    
    def generate_recommendations(self) -> List[Dict[str, str]]:
        """Original method for backward compatibility"""
        return self.generate_prioritized_recommendations()
    
    def generate_prioritized_recommendations(self) -> List[Dict[str, str]]:
        """Generate prioritized actionable recommendations with enhanced business context"""
        recommendations = []
        
        # Data quality recommendations
        missing_total = self.df.isnull().sum().sum()
        if missing_total > 0:
            recommendations.append({
                'area': 'Data Quality Improvement',
                'recommendation': 'Implement automated data validation and establish data cleaning procedures. Consider data quality monitoring dashboard.',
                'impact': 'High - ensures reliable analysis and decision making, reduces errors',
                'effort': 'Medium',
                'timeline': '1-2 months'
            })
        
        # Performance optimization based on distribution
        if len(self.numeric_cols) >= 2:
            try:
                # Analyze correlation patterns for efficiency opportunities
                corr_matrix = self.df[self.numeric_cols[:5]].corr()  # Limit for performance
                strong_corrs = (corr_matrix.abs() > 0.7) & (corr_matrix.abs() < 1.0)
                strong_pair_count = (strong_corrs.sum().sum() - len(self.numeric_cols[:5])) / 2
                
                if strong_pair_count > 0:
                    recommendations.append({
                        'area': 'Process Optimization',
                        'recommendation': f'Analyze {int(strong_pair_count)} highly correlated variable pairs to identify efficiency opportunities, process bottlenecks, and potential multicollinearity in models.',
                        'impact': 'Medium - can lead to cost savings and improved resource allocation',
                        'effort': 'Low',
                        'timeline': '2-4 weeks'
                    })
            except:
                pass
        
        # Growth and trend-based recommendations
        if self.date_cols:
            recommendations.append({
                'area': 'Growth Strategy & Forecasting',
                'recommendation': 'Implement regular trend monitoring and develop forecasting models for proactive planning. Establish performance benchmarks and early warning systems.',
                'impact': 'High - enables data-driven growth initiatives and market responsiveness',
                'effort': 'Medium',
                'timeline': '1-3 months'
            })
        
        # Customer-focused recommendations
        customer_like_cols = [
            col for col in self.categorical_cols 
            if any(term in col.lower() for term in ['customer', 'client', 'user'])
        ]
        if customer_like_cols:
            recommendations.append({
                'area': 'Customer Experience & Segmentation',
                'recommendation': 'Conduct comprehensive customer segmentation analysis and develop targeted strategies for high-value segments. Implement customer lifetime value tracking.',
                'impact': 'High - improves customer retention, lifetime value, and acquisition efficiency',
                'effort': 'Medium',
                'timeline': '2-3 months'
            })
        
        # Risk management recommendations
        outlier_insights = []
        for col in self.numeric_cols[:2]:
            try:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                    if outliers > len(self.df) * 0.05:  # More than 5% outliers
                        outlier_insights.append(col)
            except:
                continue
        
        if outlier_insights:
            recommendations.append({
                'area': 'Risk Management & Anomaly Detection',
                'recommendation': f'Investigate outliers in {", ".join(outlier_insights[:2])} and implement automated outlier detection procedures. Develop protocols for handling exceptional cases.',
                'impact': 'Medium - reduces analysis bias, identifies potential issues and opportunities',
                'effort': 'Low',
                'timeline': '3-6 weeks'
            })
        
        # Strategic planning and governance
        recommendations.append({
            'area': 'Strategic Planning & Governance',
            'recommendation': 'Establish regular business review cycles using these insights to track progress and adjust strategies. Create data governance framework and KPI monitoring system.',
            'impact': 'High - creates continuous improvement culture and strategic alignment',
            'effort': 'High',
            'timeline': '3-6 months'
        })
        
        # Performance monitoring
        recommendations.append({
            'area': 'Performance Monitoring',
            'recommendation': 'Implement dashboard for real-time performance tracking. Set up automated alerts for significant changes and anomalies.',
            'impact': 'Medium - improves responsiveness and decision speed',
            'effort': 'Medium',
            'timeline': '2-4 months'
        })
        
        # Prioritize recommendations based on impact and effort
        priority_scores = {
            'High': 3,
            'Medium': 2,
            'Low': 1
        }
        
        def calculate_priority(rec):
            impact_score = priority_scores.get(rec['impact'].split(' - ')[0], 0)
            effort_score = priority_scores.get(rec['effort'], 0)
            # Higher impact and lower effort = higher priority
            return impact_score * (4 - effort_score)  # Invert effort score
        
        return sorted(recommendations, key=calculate_priority, reverse=True)
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive executive summary with enhanced metrics"""
        summary = {
            'health_score': 0,
            'health_trend': 'Stable',
            'growth_momentum': 'Neutral',
            'risk_level': 'Medium',
            'key_highlights': [],
            'priority_actions': [],
            'performance_scorecard': {},
            'data_quality_assessment': {},
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate health score based on multiple factors
        score_components = []
        
        # Data quality score (20%)
        data_quality_score = min(self.data_quality_report['completeness_score'], 100)
        score_components.append(data_quality_score * 0.20)
        summary['data_quality_assessment']['completeness'] = data_quality_score
        
        # Performance stability (30%)
        if self.numeric_cols:
            primary_metric = self.numeric_cols[0]
            try:
                valid_values = self.df[primary_metric].dropna()
                if len(valid_values) > 0 and valid_values.mean() != 0:
                    cv = (valid_values.std() / valid_values.mean()) * 100
                    stability_score = max(0, 100 - min(cv, 100))  # Lower CV = higher stability
                    score_components.append(stability_score * 0.30)
                    summary['data_quality_assessment']['stability'] = stability_score
                else:
                    score_components.append(70 * 0.30)
            except:
                score_components.append(70 * 0.30)
        else:
            score_components.append(70 * 0.30)
        
        # Growth momentum (30%)
        growth_score = 70  # Default score
        if self.date_cols and self.numeric_cols:
            try:
                trend_data = self.analyze_trends(self.numeric_cols[0], self.date_cols[0])
                if trend_data is not None and len(trend_data) >= 2:
                    recent_growth = ((trend_data.iloc[-1] - trend_data.iloc[-2]) / abs(trend_data.iloc[-2])) * 100
                    growth_score = min(max(50 + recent_growth, 0), 100)
                    score_components.append(growth_score * 0.30)
                    
                    # Set growth momentum
                    if recent_growth > 15:
                        summary['growth_momentum'] = 'Strong ↗️'
                    elif recent_growth > 5:
                        summary['growth_momentum'] = 'Positive ↑'
                    elif recent_growth > -5:
                        summary['growth_momentum'] = 'Stable →'
                    else:
                        summary['growth_momentum'] = 'Declining ↓'
                else:
                    score_components.append(growth_score * 0.30)
            except:
                score_components.append(growth_score * 0.30)
        else:
            score_components.append(growth_score * 0.30)
        
        # Risk assessment (20%)
        risk_factors = 0
        total_factors = 0
        
        # Missing data risk
        if self.data_quality_report['missing_values'] > 0:
            risk_factors += 1
        total_factors += 1
        
        # Data quality issues risk
        if len(self.data_quality_report['quality_issues']) > 2:
            risk_factors += 1
        total_factors += 1
        
        # Outlier risk
        if self.numeric_cols:
            for col in self.numeric_cols[:2]:
                try:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > 0:
                        outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
                        if outliers > len(self.df) * 0.1:  # More than 10% outliers
                            risk_factors += 1
                    total_factors += 1
                except:
                    total_factors += 1
        
        risk_score = max(0, 100 - (risk_factors / total_factors * 100)) if total_factors > 0 else 80
        score_components.append(risk_score * 0.20)
        summary['data_quality_assessment']['risk_score'] = risk_score
        
        # Calculate overall health score
        summary['health_score'] = int(sum(score_components))
        
        # Set health trend based on components
        avg_score = sum(score_components) / len(score_components)
        if avg_score >= 80:
            summary['health_trend'] = 'Excellent ↗️'
        elif avg_score >= 70:
            summary['health_trend'] = 'Good →'
        elif avg_score >= 60:
            summary['health_trend'] = 'Fair ↘️'
        else:
            summary['health_trend'] = 'Needs Attention ↓'
        
        # Set risk level based on risk factors
        risk_ratio = risk_factors / total_factors if total_factors > 0 else 0
        if risk_ratio > 0.7:
            summary['risk_level'] = 'High 🔴'
        elif risk_ratio > 0.4:
            summary['risk_level'] = 'Medium 🟡'
        else:
            summary['risk_level'] = 'Low 🟢'
        
        # Generate key highlights
        if self.numeric_cols:
            primary_metric = self.numeric_cols[0]
            total_value = self.df[primary_metric].sum()
            avg_value = self.df[primary_metric].mean()
            
            summary['key_highlights'].append(f"Total {primary_metric}: ${total_value:,.2f}")
            summary['key_highlights'].append(f"Average {primary_metric}: ${avg_value:,.2f}")
            summary['key_highlights'].append(f"Data Quality Score: {data_quality_score:.1f}%")
            summary['key_highlights'].append(f"Business Health Score: {summary['health_score']}/100")
        
        if self.date_cols:
            try:
                min_date = self.df[self.date_cols[0]].min()
                max_date = self.df[self.date_cols[0]].max()
                summary['key_highlights'].append(f"Analysis Period: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            except:
                pass
        
        # Priority actions from recommendations
        recommendations = self.generate_prioritized_recommendations()
        for i, rec in enumerate(recommendations[:3]):  # Top 3 recommendations
            summary['priority_actions'].append({
                'action': rec['recommendation'],
                'impact': rec['impact'],
                'effort': rec['effort'],
                'timeline': rec['timeline'],
                'owner': 'Business Leadership'
            })
        
        # Performance scorecard with enhanced metrics
        summary['performance_scorecard'] = {
            'Data Quality': {
                'score': min(10, int(data_quality_score / 10)), 
                'trend': 'Stable', 
                'description': 'Data completeness & reliability'
            },
            'Business Health': {
                'score': min(10, int(summary['health_score'] / 10)), 
                'trend': summary['health_trend'], 
                'description': 'Overall business performance'
            },
            'Growth Momentum': {
                'score': min(10, int(growth_score / 10)), 
                'trend': summary['growth_momentum'], 
                'description': 'Revenue and performance trends'
            },
            'Risk Management': {
                'score': min(10, int(risk_score / 10)), 
                'trend': 'Managed', 
                'description': 'Risk control effectiveness'
            },
            'Operational Efficiency': {
                'score': 7, 
                'trend': 'Improving', 
                'description': 'Resource utilization & processes'
            }
        }
        
        return summary