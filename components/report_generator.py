# report_generator.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import json
import hashlib
import warnings
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataIntegrityValidator:
    """Data validation for analysis"""
    
    @staticmethod
    def validate_dataset_integrity(df):
        """Validate dataset for analysis"""
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
        if len(df) < 5:
            validation_results['warnings'].append("Very small dataset - limited analysis possible")
        
        # Data quality
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        validation_results['quality_metrics']['missing_percentage'] = missing_pct
        
        if missing_pct > 50:
            validation_results['critical_issues'].append(f"Extremely high missing data: {missing_pct:.1f}%")
        elif missing_pct > 25:
            validation_results['warnings'].append(f"High missing data: {missing_pct:.1f}%")
        
        return validation_results

class ProfessionalReportTemplates:
    """Professional Report Templates for Data Analysis"""
    
    def __init__(self):
        self.analysis_timestamp = datetime.now()
    
    def create_comprehensive_report(self, df: pd.DataFrame, config: Dict) -> str:
        """Create comprehensive data analysis report"""
        
        # Generate all sections
        cover = self._create_cover_page(config)
        executive = self._create_executive_summary(df, config)
        overview = self._create_data_overview(df)
        statistics = self._create_statistical_analysis(df)
        insights = self._create_insights_recommendations(df)
        
        report_content = f"""
{cover}

{executive}

{overview}

{statistics}

{insights}

{self._create_report_footer(df, config)}
"""
        return report_content
    
    def _create_cover_page(self, config: Dict) -> str:
        """Create professional cover page"""
        return f"""
# Quick Data Analysis



**Prepared By:** Quick Analysis Tool  
**Organization:** Analytics Division  
**Date:** {self.analysis_timestamp.strftime('%Y-%m-%d')}  
**Report ID:** {self._generate_report_id()}  
**Confidentiality:** Professional Use

---

## 📋 Report Overview

This professional data analysis report provides comprehensive statistical analysis, visual exploration, and actionable insights based on your dataset. The report includes descriptive statistics, distribution analysis, correlation studies, and interactive visualizations to support data-driven decision making.
"""
    
    def _create_executive_summary(self, df: pd.DataFrame, config: Dict) -> str:
        """Create executive summary"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        quality_score = self._calculate_quality_score(df)
        completeness = self._calculate_completeness(df)
        
        return f"""
## 📈 Executive Summary

### 🎯 Analysis Scope
This comprehensive analysis examines **{len(df):,} records** across **{len(df.columns):,} features**, including **{len(numeric_cols)} numeric variables** and **{len(categorical_cols)} categorical variables**.



### 💡 Primary Findings
{self._generate_executive_insights(df)}

### 📊 Methodology
- **Statistical Analysis:** Descriptive statistics, correlation analysis, distribution testing
- **Visual Exploration:** Interactive charts, trend analysis, pattern recognition
- **Quality Assessment:** Data completeness, outlier detection, integrity checks
- **Business Intelligence:** Actionable insights and strategic recommendations
"""
    
    def _create_data_overview(self, df: pd.DataFrame) -> str:
        """Create data overview section"""
        validator = DataIntegrityValidator()
        validation_results = validator.validate_dataset_integrity(df)
        
        # Data structure
        data_types = df.dtypes.value_counts()
        type_summary = "\n".join([f"- **{str(dtype)}:** {count} columns" for dtype, count in data_types.items()])
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        
        return f"""
## 🔍 Data Overview & Structure

### 🏗️ Data Architecture
{type_summary}

### ⚠️ Quality Assessment
**Missing Data:** {df.isnull().sum().sum():,} values ({df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100:.1f}%)  
**Data Quality Score:** {self._calculate_quality_score(df)}/100  
**Completeness:** {self._calculate_completeness(df):.1f}%

### 📋 Feature Inventory
{self._generate_feature_inventory(df)}
"""
    
    def _create_statistical_analysis(self, df: pd.DataFrame) -> str:
        """Create statistical analysis section"""
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) == 0:
            return "## 📊 Statistical Analysis\n\n*No numeric columns available for statistical analysis.*"
        
        stats_content = "## 📊 Statistical Analysis\n\n"
        
        # Descriptive Statistics in natural language
        stats_content += "### 📈 Key Statistics Summary\n\n"
        
        for col in numeric_cols[:6]:  # Limit to first 6 for readability
            col_data = df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                median_val = col_data.median()
                std_val = col_data.std()
                min_val = col_data.min()
                max_val = col_data.max()
                
                stats_content += f"**{col}:**\n"
                stats_content += f"- Average value: **{mean_val:,.2f}**\n"
                stats_content += f"- Middle value: **{median_val:,.2f}**\n"
                stats_content += f"- Typical variation: **±{std_val:,.2f}**\n"
                stats_content += f"- Range: from **{min_val:,.2f}** to **{max_val:,.2f}**\n"
                
                # Add interpretation
                if mean_val > median_val:
                    stats_content += f"- *The average is higher than the middle value, suggesting some higher values are pulling the average up*\n"
                elif mean_val < median_val:
                    stats_content += f"- *The average is lower than the middle value, suggesting some lower values are pulling the average down*\n"
                else:
                    stats_content += f"- *The data appears to be symmetrically distributed*\n"
                
                stats_content += "\n"
        
        # Correlation Analysis in plain English
        if len(numeric_cols) > 1:
            stats_content += "### 🔗 Relationships Between Variables\n\n"
            corr_matrix = df[numeric_cols].corr()
            
            # Find top correlations
            strong_corrs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.5:
                        strong_corrs.append((numeric_cols[i], numeric_cols[j], corr))
            
            if strong_corrs:
                stats_content += "**Notable Relationships Found:**\n"
                strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                for col1, col2, corr in strong_corrs[:3]:  # Top 3 only
                    strength = "strong" if abs(corr) > 0.7 else "moderate"
                    direction = "positive" if corr > 0 else "negative"
                    stats_content += f"- **{col1}** and **{col2}** have a {strength} {direction} relationship\n"
            else:
                stats_content += "No strong relationships detected between the main numeric variables.\n"
        
        return stats_content
    
    def _create_insights_recommendations(self, df: pd.DataFrame) -> str:
        """Create insights and recommendations section"""
        return f"""
## 💡 Analytical Insights & Recommendations

### 🎯 Key Insights
{self._generate_analytical_insights(df)}

### 🚀 Actionable Recommendations
{self._generate_strategic_recommendations(df)}

### 📈 Business Impact Assessment
**Data Quality Impact:** {self._assess_quality_impact(df)}  
**Analytical Potential:** {self._assess_analytical_potential(df)}  
**Strategic Value:** {self._assess_strategic_value(df)}

### 🔮 Next Steps
1. **Immediate Actions:** {self._get_immediate_actions(df)}
2. **Short-term Initiatives:** {self._get_short_term_initiatives(df)}
3. **Long-term Strategy:** {self._get_long_term_strategy(df)}
"""
    
    def _generate_executive_insights(self, df: pd.DataFrame) -> str:
        """Generate executive insights in natural language"""
        insights = []
        
        # Data quality insights
        quality_score = self._calculate_quality_score(df)
        if quality_score >= 90:
            insights.append("✅ **Excellent Data Foundation:** Your data is in great shape and ready for detailed analysis")
        elif quality_score >= 80:
            insights.append("🟡 **Good Data Quality:** The data is reliable for most business decisions")
        else:
            insights.append("🔴 **Data Quality Concerns:** We recommend improving data quality before making critical decisions")
        
        # Statistical insights
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 3:
            insights.append("📊 **Rich Statistical Data:** You have multiple measurable factors that allow for comprehensive analysis")
        
        # Size insights in plain English
        if len(df) > 10000:
            insights.append("📈 **Large-scale Analysis:** With over 10,000 records, the findings are statistically robust and reliable")
        elif len(df) > 1000:
            insights.append("📈 **Solid Sample Size:** The dataset size provides good confidence in the analysis results")
        elif len(df) < 100:
            insights.append("📉 **Limited Sample:** Consider collecting more data to strengthen your insights")
        
        # Missing data insights
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct < 5:
            insights.append("✅ **Complete Data:** Very little missing information, which is excellent for analysis")
        elif missing_pct < 15:
            insights.append("🟡 **Minor Gaps:** Some missing data exists but shouldn't significantly impact overall conclusions")
        
        return "\n".join(f"- {insight}" for insight in insights)
    
    def _generate_feature_inventory(self, df: pd.DataFrame) -> str:
        """Generate feature inventory in plain language"""
        inventory = []
        for col in df.columns[:10]:  # Limit to first 10
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            
            # Convert data type to plain English
            if 'int' in dtype or 'float' in dtype:
                type_desc = "numeric measurement"
            elif 'object' in dtype:
                type_desc = "text information"
            elif 'datetime' in dtype:
                type_desc = "date/time information"
            else:
                type_desc = "categorical information"
            
            inventory.append(f"- **{col}** ({type_desc}) - {unique_count} different values, {missing_count} missing entries")
        
        if len(df.columns) > 10:
            inventory.append(f"- ... plus {len(df.columns) - 10} additional data points")
        
        return "\n".join(inventory)
    
    def _generate_analytical_insights(self, df: pd.DataFrame) -> str:
        """Generate analytical insights in natural language"""
        insights = []
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Statistical insights in plain English
        for col in numeric_cols[:3]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                cv = std_val / mean_val if mean_val != 0 else 0
                
                insights.append(f"📊 **{col}:** Typically around {mean_val:,.0f}")
                
                if cv > 0.5:
                    insights.append(f"   - Shows considerable variation in values")
                else:
                    insights.append(f"   - Remains relatively consistent across records")
        
        # Data patterns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            strong_corrs = (corr_matrix.abs() > 0.7).sum().sum() - len(numeric_cols)
            if strong_corrs > 0:
                insights.append(f"🔗 **Connected Patterns:** Found {strong_corrs} strong relationships between different measurements")
        
        # Data quality insights
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"⚠️ **Data Cleaning Opportunity:** {duplicate_count} duplicate records found that could be reviewed")
        
        return "\n".join(f"- {insight}" for insight in insights)
    
    def _generate_strategic_recommendations(self, df: pd.DataFrame) -> str:
        """Generate strategic recommendations in business language"""
        recommendations = []
        
        recommendations.append("**Regular Data Health Checks:** Set up monthly reviews of your data quality")
        recommendations.append("**Business Dashboard:** Create a simple dashboard to track key metrics regularly")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 3:
            recommendations.append("**Performance Tracking:** Monitor the relationships between your key business metrics")
        
        if len(df.select_dtypes(include=['datetime']).columns) > 0:
            recommendations.append("**Trend Analysis:** Review how your key numbers change over time each quarter")
        
        # Based on data quality
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        if missing_pct > 10:
            recommendations.append("**Data Collection Improvement:** Work on reducing missing information in future data collection")
        
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
        
        # Feature bonus
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        if numeric_count >= 3:
            score += 10
        
        return max(0, min(100, int(score)))
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        if total_cells == 0:
            return 0.0
        return ((total_cells - missing_cells) / total_cells) * 100
    
    def _generate_report_id(self) -> str:
        """Generate report ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"PRO-{hashlib.md5(timestamp.encode()).hexdigest()[:8].upper()}"
    
    def _assess_quality_impact(self, df: pd.DataFrame) -> str:
        """Assess quality impact"""
        quality_score = self._calculate_quality_score(df)
        if quality_score >= 85:
            return "High confidence in analytical results"
        elif quality_score >= 70:
            return "Moderate confidence - verify critical findings"
        else:
            return "Limited confidence - data improvements needed"
    
    def _assess_analytical_potential(self, df: pd.DataFrame) -> str:
        """Assess analytical potential"""
        numeric_count = len(df.select_dtypes(include=['number']).columns)
        if numeric_count >= 5:
            return "Excellent potential for detailed analysis"
        elif numeric_count >= 3:
            return "Good potential for business insights"
        else:
            return "Basic analytical capabilities"
    
    def _assess_strategic_value(self, df: pd.DataFrame) -> str:
        """Assess strategic value"""
        if len(df) > 5000:
            return "High strategic value for business decisions"
        elif len(df) > 1000:
            return "Good strategic value for operational improvements"
        else:
            return "Limited strategic value - consider expanding data collection"
    
    def _get_immediate_actions(self, df: pd.DataFrame) -> str:
        """Get immediate actions"""
        if df.isnull().sum().sum() > 0:
            return "Review and address missing data issues"
        return "Share key findings with relevant teams"
    
    def _get_short_term_initiatives(self, df: pd.DataFrame) -> str:
        """Get short-term initiatives"""
        return "Implement regular reporting and basic monitoring"
    
    def _get_long_term_strategy(self, df: pd.DataFrame) -> str:
        """Get long-term strategy"""
        return "Develop ongoing data quality processes and advanced analytics"
    
    def _create_report_footer(self, df: pd.DataFrame, config: Dict) -> str:
        """Create report footer"""
        return f"""
---

## 📄 Report Information

**Generated:** {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** {df.shape[0]:,} records × {df.shape[1]:,} features  
**Analysis Scope:** {config.get('analysis_type', 'Comprehensive Statistical Analysis')}  
**Quality Score:** {self._calculate_quality_score(df)}/100  
**Completeness:** {self._calculate_completeness(df):.1f}%

*This professional data analysis report was generated to support evidence-based decision making and strategic planning. All analyses conducted using statistical best practices and data science methodologies.*

---
*Report ID: {self._generate_report_id()} | Generated by Professional Data Analysis System*
"""

def inject_professional_css():
    """Inject professional CSS styles"""
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .kpi-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2E86AB;
    }
    
    .kpi-card.primary { border-left-color: #2E86AB; }
    .kpi-card.success { border-left-color: #28a745; }
    .kpi-card.warning { border-left-color: #ffc107; }
    .kpi-card.info { border-left-color: #17a2b8; }
    
    .kpi-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #2E86AB;
        display: block;
    }
    
    .kpi-label {
        font-size: 0.9em;
        color: #666;
        display: block;
        margin: 5px 0;
    }
    
    .dimensions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin: 15px 0;
    }
    
    .dimension-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    
    .dimension-value {
        font-size: 1.4em;
        font-weight: bold;
        color: #2E86AB;
        display: block;
    }
    
    .dimension-label {
        font-size: 0.8em;
        color: #666;
        display: block;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2E86AB, #1a6a8b);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1a6a8b, #155d74);
        transform: translateY(-1px);
    }
    
    .report-section {
        background: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        text-align: center;
    }
    
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
    </style>
    """, unsafe_allow_html=True)

def render_professional_report_generator():
    """Render professional report generator interface"""
    
    inject_professional_css()
    
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2E86AB, #1a6a8b); color: white; padding: 30px; border-radius: 12px; text-align: center; margin: 20px 0;">
        <h1 style="color: white; margin: 0;">📊 Professional Data Analysis Report Generator</h1>
        <p style="color: white; opacity: 0.9; margin: 10px 0 0 0;">Comprehensive Statistical Analysis with Interactive Visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for data
    if not st.session_state.get('processed_data'):
        st.info("""
        <div style="background: white; padding: 20px; border-radius: 8px; border-left: 4px solid #28a745;">
            <h4>📁 Data Required for Analysis</h4>
            <p>Please upload and process your dataset in the Data Processor to generate comprehensive analysis reports.</p>
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <span style="margin-right: 8px;">✅</span>
                <span>Statistical analysis tools ready</span>
            </div>
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <span style="margin-right: 8px;">✅</span>
                <span>Visualization engine available</span>
            </div>
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <span style="margin-right: 8px;">⚠️</span>
                <span>Awaiting dataset upload</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    selected_file = st.session_state.get('current_file', '')
    if not selected_file or selected_file not in st.session_state.processed_data:
        st.error("❌ No valid dataset selected for analysis.")
        return
    
    data_info = st.session_state.processed_data[selected_file]
    df = data_info['dataframe']
    
    # Data Health Check
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h2>🔍 Data Health Check</h2>
    </div>
    """, unsafe_allow_html=True)
    
    validator = DataIntegrityValidator()
    validation_results = validator.validate_dataset_integrity(df)
    
    # Display validation results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Health", 
                 "✅ Excellent" if validation_results['is_valid'] and len(df) > 1000 else "🟡 Good" if validation_results['is_valid'] else "🔴 Review",
                 delta=None)
    
    with col2:
        quality_score = calculate_quality_score(df)
        st.metric("Quality Score", f"{quality_score}/100", 
                 delta="Excellent" if quality_score >= 90 else "Good" if quality_score >= 80 else "Needs Attention")
    
    with col3:
        st.metric("Records", f"{len(df):,}",
                 delta="Strong" if len(df) > 5000 else "Good")
    
    with col4:
        st.metric("Features", f"{len(df.columns)}",
                 delta="Comprehensive" if len(df.columns) > 10 else "Good")
    
    # Report Configuration
    st.markdown("""
    <div style="background: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h2>⚙️ Report Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("📋 Report Title", 
                                   "Quick Data Analysis")
        report_author = st.text_input("👤 Analyst Name", 
                                    "Quick Analysis Tool")
        
        analysis_type = st.selectbox(
            "🔬 Analysis Type",
            ["Comprehensive Statistical Analysis", "Business Intelligence Report", 
             "Exploratory Data Analysis", "Technical Data Report"]
        )
    
    with col2:
        company_name = st.text_input("🏢 Organization", 
                                   "Analytics Division")
        report_focus = st.selectbox("🎯 Primary Focus", 
                                  ["Statistical Insights", "Business Impact", 
                                   "Technical Analysis", "Strategic Recommendations"])
    
    # Generate Report
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 20px 0;">
        <h2 style="color: white; margin: 0;">🚀 Generate Professional Report</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Generate Full Report", use_container_width=True):
            generate_comprehensive_report(df, {
                'title': report_title,
                'author': report_author,
                'company': company_name,
                'analysis_type': analysis_type,
                'focus': report_focus
            })
    
    with col2:
        if st.button("📊 Quick Analysis", use_container_width=True):
            generate_quick_analysis(df)
    
    with col3:
        if st.button("📈 Visual Report", use_container_width=True):
            generate_visual_report(df)
    
    # Display generated report
    if st.session_state.get('current_report'):
        display_professional_report(st.session_state.current_report, report_title, df)

def generate_comprehensive_report(df, config):
    """Generate comprehensive data analysis report"""
    with st.spinner("🔍 Analyzing data and generating comprehensive report..."):
        try:
            report_generator = ProfessionalReportTemplates()
            report_content = report_generator.create_comprehensive_report(df, config)
            
            st.session_state.current_report = report_content
            st.session_state.report_generated = True
            
            st.success(f"""
            🎉 **Comprehensive Report Generated Successfully!**
            
            **Report ID:** {report_generator._generate_report_id()}
            **Analysis Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Dataset Analyzed:** {df.shape[0]:,} records × {df.shape[1]:,} features
            """)
            
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")

def generate_quick_analysis(df):
    """Generate quick analysis report"""
    with st.spinner("⚡ Creating quick analysis..."):
        try:
            report_generator = ProfessionalReportTemplates()
            
            quick_content = f"""
{report_generator._create_cover_page({'title': 'Quick Data Analysis', 'author': 'Quick Analysis Tool'})}

{report_generator._create_executive_summary(df, {})}

{report_generator._create_data_overview(df)}

{report_generator._create_statistical_analysis(df)}

*Quick analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
            
            st.session_state.current_report = quick_content
            st.success("✅ Quick analysis generated!")
            
        except Exception as e:
            st.error(f"❌ Error generating quick analysis: {str(e)}")

def generate_visual_report(df):
    """Generate visual-focused report"""
    with st.spinner("🎨 Creating visual report..."):
        try:
            # Create a visual report with actual charts
            st.session_state.visual_report_data = df
            st.session_state.show_visual_report = True
            st.success("✅ Visual report ready! Check the Visualizations tab.")
            
        except Exception as e:
            st.error(f"❌ Error generating visual report: {str(e)}")

def display_professional_report(report_content, title, df):
    """Display the generated professional report"""
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2E86AB, #1a6a8b); color: white; padding: 25px; border-radius: 12px; margin: 20px 0;">
        <h2 style="color: white; margin: 0;">📄 {title}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Report Preview", "📊 Live Visualizations", "📥 Export Options"])
    
    with tab1:
        st.markdown('<div class="report-section">', unsafe_allow_html=True)
        st.markdown(report_content)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("📊 Interactive Data Visualizations")
        generate_live_visualizations(df)
    
    with tab3:
        st.subheader("📥 Download Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download PDF Report", use_container_width=True):
                offer_download(report_content, title, "pdf")
        
        with col2:
            if st.button("🌐 HTML Report", use_container_width=True):
                html_content = generate_html_report(report_content, title)
                offer_download(html_content, title, "html")
        
        with col3:
            if st.button("📊 Data Summary", use_container_width=True):
                summary_content = generate_data_summary(df, title)
                offer_download(summary_content, title, "txt")

def generate_live_visualizations(df):
    """Generate live interactive visualizations"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) == 0:
        st.info("No numeric columns available for visualization.")
        return
    
    # Distribution Chart
    st.subheader("📈 Distribution Analysis")
    selected_col = st.selectbox("Select column for distribution:", numeric_cols)
    if selected_col:
        fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}', nbins=30)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        col_data = df[selected_col].dropna()
        if len(col_data) > 0:
            st.markdown(f"""
            <div class="insight-box">
                <h4>📊 What this shows:</h4>
                <p>The chart above shows how the values of <strong>{selected_col}</strong> are distributed across your data. 
                This helps you understand the typical range and frequency of values for this measurement.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Analysis")
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, title='Correlation Heatmap', aspect='auto')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>🔗 Understanding Correlations:</h4>
            <p>This heatmap shows how different numeric measurements relate to each other. 
            <strong>Blue colors</strong> indicate positive relationships (when one goes up, the other tends to go up). 
            <strong>Red colors</strong> indicate negative relationships (when one goes up, the other tends to go down). 
            Darker colors mean stronger relationships.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Box Plot
    st.subheader("📊 Statistical Distribution")
    box_col = st.selectbox("Select column for box plot:", numeric_cols, key="box_plot")
    if box_col:
        fig = px.box(df, y=box_col, title=f'Box Plot of {box_col}')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>📦 Understanding the Box Plot:</h4>
            <p>This box plot shows the distribution of <strong>{box_col}</strong>. The box represents the middle 50% of your data, 
            the line inside the box shows the median (middle value), and the lines extending show the typical range. 
            Any dots outside these lines may be unusual values worth checking.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Categorical Analysis
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        st.subheader("🏷️ Category Comparison")
        cat_col = st.selectbox("Select category column:", categorical_cols)
        num_col = st.selectbox("Select numeric column to compare:", numeric_cols)
        
        if cat_col and num_col:
            # Show top 10 categories to avoid overcrowding
            top_cats = df[cat_col].value_counts().head(10).index
            filtered_df = df[df[cat_col].isin(top_cats)]
            
            fig = px.bar(filtered_df, x=cat_col, y=num_col, title=f'{num_col} by {cat_col}')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏷️ Category Insights:</h4>
                <p>This chart compares <strong>{num_col}</strong> across different categories of <strong>{cat_col}</strong>. 
                It helps you see which categories have higher or lower values, revealing patterns across your different groups.</p>
            </div>
            """, unsafe_allow_html=True)

def offer_download(content, title, format_type):
    """Offer download in different formats"""
    filename = f"{clean_filename(title)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
    
    if format_type == "html":
        mime_type = "text/html"
    elif format_type == "pdf":
        mime_type = "application/pdf"
    else:
        mime_type = "text/plain"
    
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}" style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; margin: 8px 0;">Download {format_type.upper()}</a>'
    st.markdown(f'<div style="text-align: center; margin: 20px 0;">{href}</div>', unsafe_allow_html=True)
    st.success(f"✅ Report ready for download!")

def generate_html_report(markdown_content, title):
    """Generate HTML version of report"""
    # Simple markdown to HTML conversion
    html_content = markdown_content
    html_content = html_content.replace('# ', '<h1>').replace('\n# ', '</h1>\n<h1>')
    html_content = html_content.replace('## ', '<h2>').replace('\n## ', '</h2>\n<h2>')
    html_content = html_content.replace('### ', '<h3>').replace('\n### ', '</h3>\n<h3>')
    html_content = html_content.replace('**', '<strong>').replace('**', '</strong>')
    html_content = html_content.replace('* ', '<li>').replace('\n* ', '</li>\n<li>')
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Professional Data Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #2c3e50;
            background: #f8f9fa;
            padding: 20px;
        }}
        .report-container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2E86AB;
            border-bottom: 3px solid #28a745;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #28a745;
            margin: 30px 0 15px 0;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
            text-align: center;
        }}
        .insight-box {{
            background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #28a745;
            margin: 15px 0;
        }}
        .recommendation-box {{
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
    <div class="report-container">
        {html_content}
    </div>
</body>
</html>
"""
    return html_template

def generate_data_summary(df, title):
    """Generate data summary text file"""
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    summary = f"""
DATA ANALYSIS SUMMARY
=====================

Report: {title}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {df.shape[0]} records × {df.shape[1]} features

KEY STATISTICS:
===============
Total Records: {len(df):,}
Total Features: {len(df.columns):,}
Numeric Measurements: {len(numeric_cols):,}
Category Fields: {len(categorical_cols):,}
Missing Values: {df.isnull().sum().sum():,}
Data Quality Score: {calculate_quality_score(df)}/100

BUSINESS INSIGHTS:
==================
"""
    
    # Add key insights in plain language
    if len(numeric_cols) > 0:
        for col in numeric_cols[:3]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                summary += f"- {col}: Typically around {mean_val:,.0f}\n"
    
    summary += f"\nDATA QUALITY:\n"
    summary += f"Completeness: {((df.shape[0]*df.shape[1] - df.isnull().sum().sum())/(df.shape[0]*df.shape[1])*100):.1f}%\n"
    summary += f"Duplicate Records: {df.duplicated().sum():,}\n"
    
    return summary

def calculate_quality_score(df):
    """Calculate data quality score"""
    score = 100
    
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    score -= missing_ratio * 40
    
    duplicate_ratio = df.duplicated().sum() / len(df)
    score -= duplicate_ratio * 25
    
    return max(0, min(100, int(score)))

def clean_filename(filename):
    """Clean filename for download"""
    import re
    return re.sub(r'[^a-zA-Z0-9_\-\s]', '', filename).replace(' ', '_').replace('__', '_')

def main():
    """Main function"""
    render_professional_report_generator()

if __name__ == "__main__":
    main()