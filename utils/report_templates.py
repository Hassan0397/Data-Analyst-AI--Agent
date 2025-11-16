# report_templates.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import hashlib
import scipy.stats as stats
from scipy import stats as scipy_stats
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import base64
import io

class ReportTemplates:
    """Professional Report Templates with Complete Data Analysis & Visualizations"""
    
    def __init__(self):
        self.analysis_timestamp = datetime.now()
        self.validator = DataIntegrityValidator()
    
    def create_comprehensive_analysis_report(self, df: pd.DataFrame, title: str, author: str, **kwargs) -> str:
        """Create comprehensive data analysis report with visuals"""
        company = kwargs.get('company', 'Data Analytics Division')
        
        # Generate all analysis components
        executive_summary = self.create_executive_summary(df, title)
        data_overview = self.create_data_overview(df)
        statistical_analysis = self.create_statistical_report(df)
        visualizations = self.create_visualizations_section(df)
        insights = self.create_business_insights(df)
        
        return f"""
# Quick Data Analysis

## {title}

<div style="background: linear-gradient(135deg, #2E86AB, #1a6a8b); color: white; padding: 30px; border-radius: 12px; text-align: center; margin: 20px 0;">
    <h2 style="color: white; margin: 0;">Quick Data Analysis</h2>
    <p style="color: white; opacity: 0.9; margin: 10px 0 0 0;">Comprehensive Statistical Analysis & Visual Insights</p>
</div>

### Report Details
**Analyst:** {author}  
**Organization:** {company}  
**Analysis Date:** {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Report ID:** {self._generate_report_id(title)}  
**Dataset:** {df.shape[0]:,} records × {df.shape[1]:,} features

---

{executive_summary}

---

{data_overview}

---

{statistical_analysis}

---

{visualizations}

---

{insights}

---

## Conclusion & Next Steps

This comprehensive analysis provides a complete view of your dataset with statistical insights and visual exploration. Key patterns and relationships have been identified to support data-driven decision making.

**Recommendations:**
- Monitor key metrics identified in statistical analysis
- Use visualizations for ongoing performance tracking
- Consider additional analysis based on identified patterns
- Implement automated reporting for regular updates

*Report generated using professional data analysis standards on {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}*
"""
    
    def create_executive_summary(self, df: pd.DataFrame, title: str) -> str:
        """Create executive summary with key findings"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Key statistics
        total_records = len(df)
        completeness = self._calculate_completeness(df)
        quality_score = self._calculate_quality_score(df)
        
        # Top insights
        insights = self._generate_key_insights(df)
        
        return f"""
## Executive Summary

### Analysis Overview
This report provides a comprehensive analysis of **{title}**, examining **{total_records:,} records** across **{len(df.columns):,} features**. The dataset includes **{len(numeric_cols)} numeric variables** and **{len(categorical_cols)} categorical variables**.

### Key Performance Indicators

<div class="kpi-grid">
    <div class="kpi-card primary">
        <span class="kpi-value">{total_records:,}</span>
        <span class="kpi-label">Total Records</span>
    </div>
    <div class="kpi-card success">
        <span class="kpi-value">{completeness:.1f}%</span>
        <span class="kpi-label">Data Completeness</span>
    </div>
    <div class="kpi-card warning">
        <span class="kpi-value">{quality_score}/100</span>
        <span class="kpi-label">Quality Score</span>
    </div>
    <div class="kpi-card info">
        <span class="kpi-value">{len(numeric_cols)}</span>
        <span class="kpi-label">Numeric Features</span>
    </div>
</div>

### Primary Findings
{insights}

### Methodology
- **Statistical Analysis:** Descriptive statistics, correlation analysis, distribution testing
- **Visual Exploration:** Interactive charts, trend analysis, pattern recognition
- **Quality Assessment:** Data completeness, outlier detection, integrity checks
- **Business Intelligence:** Actionable insights and strategic recommendations
"""
    
    def create_data_overview(self, df: pd.DataFrame) -> str:
        """Create comprehensive data overview"""
        validation_results = self.validator.validate_dataset_integrity(df)
        
        # Data structure analysis
        data_types = df.dtypes.value_counts()
        type_summary = "\n".join([f"- **{str(dtype)}:** {count} columns" for dtype, count in data_types.items()])
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        
        # Missing data analysis
        missing_by_col = df.isnull().sum()
        missing_cols = missing_by_col[missing_by_col > 0]
        missing_summary = "\n".join([f"- **{col}:** {missing_by_col[col]:,} missing ({missing_by_col[col]/len(df)*100:.1f}%)" 
                                   for col in missing_cols.index[:5]])  # Top 5 only
        
        if missing_summary == "":
            missing_summary = "- No missing values detected"
        
        return f"""
## Data Overview & Structure

### Dataset Dimensions
<div class="dimensions-grid">
    <div class="dimension-card">
        <span class="dimension-value">{df.shape[0]:,}</span>
        <span class="dimension-label">Records</span>
    </div>
    <div class="dimension-card">
        <span class="dimension-value">{df.shape[1]:,}</span>
        <span class="dimension-label">Features</span>
    </div>
    <div class="dimension-card">
        <span class="dimension-value">{memory_usage:.1f} MB</span>
        <span class="dimension-label">Memory</span>
    </div>
    <div class="dimension-card">
        <span class="dimension-value">{df.duplicated().sum():,}</span>
        <span class="dimension-label">Duplicates</span>
    </div>
</div>

### Data Architecture
{type_summary}

### Quality Assessment
**Missing Data:** {df.isnull().sum().sum():,} values ({df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)  
**Data Quality Score:** {self._calculate_quality_score(df)}/100  
**Completeness:** {self._calculate_completeness(df):.1f}%

### Feature Inventory
{self._generate_column_summary(df)}
"""
    
    def create_statistical_report(self, df: pd.DataFrame) -> str:
        """Create comprehensive statistical analysis"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) == 0:
            return "## Statistical Analysis\n\nNo numeric columns available for statistical analysis."
        
        stats_content = "## Statistical Analysis\n\n"
        
        # Descriptive Statistics
        stats_content += "### Key Statistics Summary\n\n"
        desc_stats = df[numeric_cols].describe()
        for col in numeric_cols[:6]:  # Limit to first 6 columns
            col_data = df[col].dropna()
            if len(col_data) > 0:
                stats_content += f"**{col}:**\n"
                stats_content += f"- Count: {desc_stats[col]['count']:,}\n"
                stats_content += f"- Mean: {desc_stats[col]['mean']:.2f}\n"
                stats_content += f"- Std: {desc_stats[col]['std']:.2f}\n"
                stats_content += f"- Min: {desc_stats[col]['min']:.2f}\n"
                stats_content += f"- 25%: {desc_stats[col]['25%']:.2f}\n"
                stats_content += f"- 50% (Median): {desc_stats[col]['50%']:.2f}\n"
                stats_content += f"- 75%: {desc_stats[col]['75%']:.2f}\n"
                stats_content += f"- Max: {desc_stats[col]['max']:.2f}\n\n"
        
        # Correlation Analysis
        if len(numeric_cols) > 1:
            stats_content += "### Relationships Between Variables\n\n"
            corr_matrix = df[numeric_cols].corr()
            strong_correlations = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.7:
                        strong_correlations.append((numeric_cols[i], numeric_cols[j], corr))
            
            if strong_correlations:
                stats_content += "**Strong Correlations Found:**\n"
                for col1, col2, corr in strong_correlations[:5]:  # Limit to top 5
                    stats_content += f"- **{col1}** ↔ **{col2}**: {corr:.3f}\n"
                stats_content += "\n"
            else:
                stats_content += "No strong relationships detected between the main numeric variables.\n\n"
        
        # Distribution Analysis
        stats_content += "### Distribution Analysis\n\n"
        for col in numeric_cols[:4]:  # Limit to first 4 columns
            col_data = df[col].dropna()
            if len(col_data) > 0:
                skewness = col_data.skew()
                kurtosis = col_data.kurtosis()
                stats_content += f"**{col} Distribution:**\n"
                stats_content += f"- Skewness: {skewness:.3f} ({'symmetric' if abs(skewness) < 0.5 else 'moderately skewed' if abs(skewness) < 1 else 'highly skewed'})\n"
                stats_content += f"- Kurtosis: {kurtosis:.3f} ({'normal' if abs(kurtosis) < 0.5 else 'heavy-tailed' if kurtosis > 0.5 else 'light-tailed'})\n\n"
        
        return stats_content
    
    def create_visualizations_section(self, df: pd.DataFrame) -> str:
        """Create visualizations section with embedded charts"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        visuals_content = "## Data Visualizations\n\n"
        
        # Generate visualizations
        charts = self._generate_all_visualizations(df)
        
        # Add chart descriptions and insights
        visuals_content += "### Trend Analysis\n"
        if 'trend_chart' in charts:
            visuals_content += f"![Trend Chart]({charts['trend_chart']})\n\n"
            visuals_content += "*Time series analysis showing patterns and trends over the dataset.*\n\n"
        
        visuals_content += "### Distribution Analysis\n"
        if 'distribution_chart' in charts:
            visuals_content += f"![Distribution Chart]({charts['distribution_chart']})\n\n"
            visuals_content += "*Data distribution analysis showing value frequencies and patterns.*\n\n"
        
        visuals_content += "### Comparative Analysis\n"
        if 'bar_chart' in charts:
            visuals_content += f"![Bar Chart]({charts['bar_chart']})\n\n"
            visuals_content += "*Comparative analysis across different categories and segments.*\n\n"
        
        visuals_content += "### Correlation Heatmap\n"
        if 'correlation_heatmap' in charts:
            visuals_content += f"![Correlation Heatmap]({charts['correlation_heatmap']})\n\n"
            visuals_content += "*Relationship analysis between numeric variables.*\n\n"
        
        visuals_content += "### Box Plot Analysis\n"
        if 'box_plot' in charts:
            visuals_content += f"![Box Plot]({charts['box_plot']})\n\n"
            visuals_content += "*Statistical distribution and outlier detection.*\n\n"
        
        return visuals_content
    
    def create_business_insights(self, df: pd.DataFrame) -> str:
        """Create business insights and recommendations"""
        insights = self._generate_comprehensive_insights(df)
        recommendations = self._generate_data_recommendations(df)
        
        return f"""
## Analytical Insights & Recommendations

### Key Insights
{insights}

### Recommended Actions
{recommendations}

### Business Impact Assessment
**Data Quality Impact:** {self._assess_data_quality_impact(df)}
**Analytical Potential:** {self._assess_analytical_potential(df)}
**Strategic Value:** {self._assess_business_impact(df)}
"""
    
    def _generate_all_visualizations(self, df: pd.DataFrame) -> Dict[str, str]:
        """Generate all types of visualizations"""
        charts = {}
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        try:
            # Trend Chart (if datetime data available)
            datetime_cols = df.select_dtypes(include=['datetime']).columns
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                date_col = datetime_cols[0]
                value_col = numeric_cols[0]
                fig = px.line(df, x=date_col, y=value_col, title=f'Trend Analysis: {value_col} over Time')
                charts['trend_chart'] = self._fig_to_base64(fig)
            
            # Distribution Chart (Histogram)
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(df, x=col, title=f'Distribution of {col}', nbins=30)
                charts['distribution_chart'] = self._fig_to_base64(fig)
            
            # Bar Chart
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                # Take top 10 categories to avoid overcrowding
                top_categories = df[cat_col].value_counts().head(10).index
                filtered_df = df[df[cat_col].isin(top_categories)]
                fig = px.bar(filtered_df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
                charts['bar_chart'] = self._fig_to_base64(fig)
            
            # Donut Chart
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                value_counts = df[cat_col].value_counts().head(8)  # Top 8 categories
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f'Distribution of {cat_col}', hole=0.4)
                charts['donut_chart'] = self._fig_to_base64(fig)
            
            # Correlation Heatmap
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title='Correlation Heatmap', aspect='auto')
                charts['correlation_heatmap'] = self._fig_to_base64(fig)
            
            # Box Plot
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.box(df, y=col, title=f'Box Plot of {col}')
                charts['box_plot'] = self._fig_to_base64(fig)
                
        except Exception as e:
            # Fallback simple visualization if complex ones fail
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                charts['fallback_chart'] = self._fig_to_base64(fig)
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string"""
        try:
            img_bytes = fig.to_image(format="png", width=800, height=400)
            base64_str = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{base64_str}"
        except:
            return ""
    
    def _generate_key_insights(self, df: pd.DataFrame) -> str:
        """Generate key insights from data"""
        insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Data quality insights
        completeness = self._calculate_completeness(df)
        if completeness >= 95:
            insights.append("✅ **Excellent Data Quality:** Dataset is highly complete and reliable for analysis")
        elif completeness >= 85:
            insights.append("🟡 **Good Data Quality:** Minor data quality issues detected but suitable for analysis")
        else:
            insights.append("🔴 **Data Quality Concerns:** Significant missing data may affect analysis reliability")
        
        # Size insights
        if len(df) > 10000:
            insights.append("📊 **Large Dataset:** Substantial data volume supports robust statistical analysis")
        elif len(df) > 1000:
            insights.append("📊 **Moderate Dataset:** Good sample size for reliable insights")
        else:
            insights.append("📊 **Small Dataset:** Consider collecting more data for stronger conclusions")
        
        # Feature insights
        if len(numeric_cols) >= 5:
            insights.append("🔢 **Rich Numeric Data:** Multiple numeric variables enable comprehensive statistical analysis")
        if len(categorical_cols) >= 3:
            insights.append("🏷️ **Good Categorical Coverage:** Multiple categories support segmentation analysis")
        
        # Pattern insights
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            strong_corrs = (corr_matrix.abs() > 0.7).sum().sum() - len(numeric_cols)  # Exclude diagonal
            if strong_corrs > 0:
                insights.append(f"🔗 **Strong Relationships:** {strong_corrs} strong correlations detected between variables")
        
        return "\n".join(f"- {insight}" for insight in insights[:6])  # Limit to 6 insights
    
    def _generate_column_summary(self, df: pd.DataFrame) -> str:
        """Generate column-by-column summary"""
        summary = []
        for col in df.columns[:10]:  # Limit to first 10 columns
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            unique = df[col].nunique()
            missing = df[col].isnull().sum()
            
            summary.append(f"**{col}** ({dtype}) - {non_null:,} non-null, {unique:,} unique, {missing:,} missing")
        
        if len(df.columns) > 10:
            summary.append(f"... and {len(df.columns) - 10} more columns")
        
        return "\n".join(summary)
    
    def _generate_comprehensive_insights(self, df: pd.DataFrame) -> str:
        """Generate comprehensive insights"""
        insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Statistical insights
        for col in numeric_cols[:3]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                
                if abs(mean_val - median_val) > std_val * 0.5:
                    insights.append(f"📊 **{col}** shows skewness (mean: {mean_val:.2f}, median: {median_val:.2f})")
                else:
                    insights.append(f"📊 **{col}** is relatively symmetric (mean: {mean_val:.2f}, median: {median_val:.2f})")
                
                # Outlier detection
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = col_data[(col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))]
                if len(outliers) > 0:
                    insights.append(f"⚠️ **{col}** has {len(outliers):,} potential outliers")
        
        # Data quality insights
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 20:
            insights.append("🔴 **High Missing Data:** Consider data imputation strategies")
        elif missing_pct > 10:
            insights.append("🟡 **Moderate Missing Data:** Some analysis may be affected")
        
        return "\n".join(f"- {insight}" for insight in insights)
    
    def _generate_data_recommendations(self, df: pd.DataFrame) -> str:
        """Generate data recommendations"""
        recommendations = []
        
        # Always include these
        recommendations.append("**Regular Monitoring:** Implement ongoing data quality checks")
        recommendations.append("**Automated Reporting:** Set up automated analysis pipelines")
        
        # Data quality recommendations
        if df.isnull().sum().sum() > 0:
            recommendations.append("**Data Cleaning:** Address missing values through imputation or collection")
        
        if df.duplicated().sum() > 0:
            recommendations.append("**Duplicate Management:** Implement duplicate detection and removal processes")
        
        # Analysis recommendations
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 3:
            recommendations.append("**Advanced Analytics:** Consider regression analysis and predictive modeling")
        
        if len(df.select_dtypes(include=['datetime']).columns) > 0:
            recommendations.append("**Time Series Analysis:** Implement trend forecasting and seasonal analysis")
        
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> int:
        """Calculate data quality score"""
        score = 100
        
        # Missing data penalty
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        score -= missing_ratio * 40
        
        # Duplicate penalty
        duplicate_ratio = df.duplicated().sum() / len(df)
        score -= duplicate_ratio * 25
        
        # Feature diversity bonus
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_count >= 3 and categorical_count >= 2:
            score += 10
        
        # Size bonus
        if len(df) > 10000:
            score += 10
        elif len(df) > 1000:
            score += 5
        
        return max(0, min(100, int(score)))
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness percentage"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        if total_cells == 0:
            return 0.0
        return ((total_cells - missing_cells) / total_cells) * 100
    
    def _generate_report_id(self, title: str) -> str:
        """Generate report ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        content_hash = hashlib.sha256(f"{title}{timestamp}".encode()).hexdigest()[:8].upper()
        return f"PRO-{content_hash}"
    
    def _assess_data_quality_impact(self, df: pd.DataFrame) -> str:
        """Assess data quality impact"""
        quality_score = self._calculate_quality_score(df)
        if quality_score >= 90:
            return "High reliability for decision-making"
        elif quality_score >= 80:
            return "Good reliability for most analyses"
        else:
            return "Limited reliability - verify critical findings"
    
    def _assess_analytical_potential(self, df: pd.DataFrame) -> str:
        """Assess analytical potential"""
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_count >= 5 and categorical_count >= 3:
            return "Excellent potential for advanced analytics"
        elif numeric_count >= 3:
            return "Good potential for statistical analysis"
        else:
            return "Basic analytical capabilities"
    
    def _assess_business_impact(self, df: pd.DataFrame) -> str:
        """Assess business impact"""
        if len(df) > 5000 and len(df.columns) >= 8:
            return "High impact potential for strategic decisions"
        elif len(df) > 1000:
            return "Moderate impact for operational improvements"
        else:
            return "Limited impact - consider data expansion"

class DataIntegrityValidator:
    """Data integrity validation"""
    
    @staticmethod
    def validate_dataset_integrity(df: pd.DataFrame) -> Dict:
        """Validate dataset integrity"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'critical_issues': [],
            'quality_metrics': {}
        }
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['critical_issues'].append("Dataset is empty")
            return validation_results
        
        # Basic validation
        if len(df) < 10:
            validation_results['warnings'].append("Small dataset may limit analysis reliability")
        
        if len(df.columns) == 0:
            validation_results['critical_issues'].append("No columns available for analysis")
            validation_results['is_valid'] = False
        
        # Data quality checks
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        validation_results['quality_metrics']['missing_percentage'] = missing_pct
        
        if missing_pct > 30:
            validation_results['critical_issues'].append(f"High missing data: {missing_pct:.1f}%")
        elif missing_pct > 15:
            validation_results['warnings'].append(f"Moderate missing data: {missing_pct:.1f}%")
        
        return validation_results

# Professional CSS styles
PROFESSIONAL_CSS = """
<style>
.kpi-grid, .dimensions-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.kpi-card, .dimension-card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #2E86AB;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    text-align: center;
    transition: transform 0.2s ease;
}

.kpi-card:hover, .dimension-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

.kpi-value, .dimension-value {
    font-size: 1.8em;
    font-weight: bold;
    color: #2E86AB;
    display: block;
}

.kpi-label, .dimension-label {
    font-size: 0.9em;
    color: #666;
    display: block;
    margin: 5px 0;
}

.kpi-card.primary { border-left-color: #2E86AB; }
.kpi-card.success { border-left-color: #28a745; }
.kpi-card.warning { border-left-color: #ffc107; }
.kpi-card.info { border-left-color: #17a2b8; }

.insight-box {
    background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin: 15px 0;
}

.recommendation-box {
    background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    padding: 20px;
    border-radius: 8px;
    border-left: 4px solid #2196f3;
    margin: 15px 0;
}

.chart-container {
    background: white;
    padding: 20px;
    border-radius: 8px;
    margin: 20px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.chart-container img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
}
</style>
"""

def get_professional_css():
    """Return professional CSS styles"""
    return PROFESSIONAL_CSS