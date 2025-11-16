import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, normaltest, anderson
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Any, Optional, Union
import io
import base64
from utils.statistical_tests import AdvancedStatisticalTests

# Configuration
st.set_page_config(page_title="Advanced Statistical Analysis", layout="wide")

def render_statistical_analysis():
    """Enhanced statistical analysis interface with professional features and user guides"""
    
    st.header("🔬 Advanced Statistical Analysis Platform")
    
    # Quick Start Guide
    with st.expander("🚀 **Quick Start Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Welcome to the Statistical Analysis Platform!
        
        **For Beginners:**
        1. **Start with Data Overview** - Understand your dataset first
        2. **Check Normality** - See if your data follows normal distribution
        3. **Use Test Selection Helper** - Get guidance on which test to use
        4. **Follow Step-by-Step Guides** - Each section has expandable guides
        
        **For Experts:**
        - Use advanced features like Bayesian tests, Time Series, and Survival analysis
        - Batch processing for multiple analyses
        - Export comprehensive reports
        
        **Key Features:**
        - ✅ **Assumption checking** for all tests
        - ✅ **Effect size calculations** with interpretations
        - ✅ **Power analysis** for sample planning
        - ✅ **Interactive visualizations**
        - ✅ **Export capabilities** for reports
        """)
    
    if not st.session_state.processed_data:
        st.warning("📁 No data available. Please upload files first.")
        return
    
    selected_file = st.session_state.current_file
    data_info = st.session_state.processed_data[selected_file]
    df = data_info['dataframe']
    statistical_tester = AdvancedStatisticalTests(df)
    
    # Test Selection Helper
    with st.sidebar:
        st.header("🎯 Test Selection Helper")
        
        analysis_goal = st.selectbox(
            "What do you want to analyze?",
            ["Compare groups", "Check relationships", "Test assumptions", "Time series", "Survival analysis", "Advanced modeling"]
        )
        
        if analysis_goal == "Compare groups":
            st.info("""
            **Recommended Tests:**
            - **2 Groups**: T-test, Mann-Whitney
            - **3+ Groups**: ANOVA, Kruskal-Wallis
            - **Before/After**: Paired t-test, Wilcoxon
            """)
        elif analysis_goal == "Check relationships":
            st.info("""
            **Recommended Tests:**
            - **Continuous variables**: Correlation, Regression
            - **Categorical variables**: Chi-square test
            - **Multiple variables**: Factor analysis
            """)
        elif analysis_goal == "Test assumptions":
            st.info("""
            **Start with:**
            - Normality tests
            - Data overview
            - Correlation analysis
            """)
    
    # Professional dashboard layout
    st.sidebar.header("🔧 Analysis Configuration")
    
    # Export functionality
    st.sidebar.header("📤 Export Results")
    if st.sidebar.button("Export Analysis History"):
        try:
            export_data = statistical_tester.export_analysis_history('json')
            st.sidebar.download_button(
                label="Download JSON Report",
                data=export_data,
                file_name=f"statistical_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")
    
    # Statistical analysis tabs
    stat_tabs = st.tabs([
        "📊 Data Overview",
        "📈 Normality & Diagnostics", 
        "🔍 Hypothesis Testing",
        "📋 ANOVA & Group Analysis",
        "⚡ Non-Parametric Tests",
        "🔄 Bayesian Analysis",
        "⏰ Time Series",
        "📉 Survival Analysis",
        "🌐 Multivariate Analysis",
        "🎯 Power Analysis",
        "📋 Comprehensive Summary"
    ])
    
    with stat_tabs[0]:
        render_data_overview(statistical_tester)
    
    with stat_tabs[1]:
        render_enhanced_normality_tests(statistical_tester)
    
    with stat_tabs[2]:
        render_enhanced_hypothesis_testing(statistical_tester)
    
    with stat_tabs[3]:
        render_enhanced_anova_tests(statistical_tester)
    
    with stat_tabs[4]:
        render_enhanced_nonparametric_tests(statistical_tester)
    
    with stat_tabs[5]:
        render_bayesian_analysis(statistical_tester)
    
    with stat_tabs[6]:
        render_time_series_analysis(statistical_tester)
    
    with stat_tabs[7]:
        render_survival_analysis(statistical_tester)
    
    with stat_tabs[8]:
        render_multivariate_analysis(statistical_tester)
    
    with stat_tabs[9]:
        render_power_analysis(statistical_tester)
    
    with stat_tabs[10]:
        render_comprehensive_summary(statistical_tester)

def render_data_overview(statistical_tester):
    """Render comprehensive data overview with guides"""
    
    with st.expander("📚 **Data Overview Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Your Data
        
        **Why Start Here?**
        - Identify data types and structure
        - Spot missing values and data quality issues
        - Understand variable distributions
        - Plan appropriate statistical tests
        
        **What to Look For:**
        - 🔍 **Missing Data**: >20% missing may need imputation
        - 📊 **Data Types**: Numeric vs Categorical variables
        - 📈 **Distributions**: Normal vs skewed distributions
        - 🔗 **Correlations**: Relationships between variables
        """)
    
    st.subheader("📊 Dataset Overview")
    
    df = statistical_tester.df
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", len(df))
    with col2:
        st.metric("Numerical Variables", len(statistical_tester.numeric_cols))
    with col3:
        st.metric("Categorical Variables", len(statistical_tester.categorical_cols))
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    # Data preview
    st.write("#### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data types information
    st.write("#### Data Types Summary")
    dtype_summary = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df)) * 100
    })
    st.dataframe(dtype_summary, use_container_width=True)
    
    # Batch data quality check
    if st.button("🔍 Run Batch Data Quality Check"):
        with st.spinner("Analyzing data quality..."):
            # Check for high missing columns
            high_missing = dtype_summary[dtype_summary['Null Percentage'] > 20]
            if not high_missing.empty:
                st.warning(f"⚠️ **High Missing Data**: {len(high_missing)} columns have >20% missing values")
            
            # Check for constant columns
            constant_cols = []
            for col in statistical_tester.numeric_cols:
                if df[col].nunique() == 1:
                    constant_cols.append(col)
            
            if constant_cols:
                st.warning(f"⚠️ **Constant Columns**: {constant_cols} have no variation")

def render_enhanced_normality_tests(statistical_tester):
    """Enhanced normality tests with comprehensive diagnostics and guides"""
    
    with st.expander("📚 **Normality Tests Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Normality Tests
        
        **Why Test for Normality?**
        - Many statistical tests assume normal distribution
        - Helps choose between parametric vs non-parametric tests
        - Identifies data transformation needs
        
        **Key Tests:**
        - **Shapiro-Wilk**: Best for small samples (<5000)
        - **D'Agostino**: Tests skewness and kurtosis
        - **Anderson-Darling**: Good for all sample sizes
        
        **Interpretation:**
        - **p > 0.05**: Data appears normal
        - **p ≤ 0.05**: Data may not be normal
        - **Check Q-Q Plot**: Points should follow the line
        
        **If Not Normal:**
        - Use non-parametric tests
        - Consider data transformation
        - Use larger sample sizes
        """)
    
    st.subheader("📈 Normality Tests & Distribution Diagnostics")
    
    df = statistical_tester.df
    numeric_cols = statistical_tester.numeric_cols
    
    if not numeric_cols:
        st.info("📊 No numeric columns found for normality testing")
        return
    
    # Batch normality testing option
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_col = st.selectbox("Select column for normality analysis", numeric_cols, key="normality_col")
    with col2:
        if st.button("📋 Batch Test All"):
            with st.spinner("Running batch normality tests..."):
                batch_results = statistical_tester.batch_normality_tests(numeric_cols[:5])  # Limit to first 5
                st.write("#### Batch Normality Results")
                for col, result in batch_results.items():
                    if 'overall_normality' in result:
                        is_normal = result['overall_normality'].get('is_normal', False)
                        status = "✅ Normal" if is_normal else "❌ Not Normal"
                        st.write(f"**{col}**: {status}")
    
    if selected_col:
        col_data = df[selected_col].dropna()
        
        # Comprehensive distribution analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced histogram with distribution curve
            fig_hist = px.histogram(
                col_data, 
                title=f"Distribution of {selected_col}",
                nbins=50,
                marginal="box"
            )
            
            # Add normal distribution curve
            x_norm = np.linspace(col_data.min(), col_data.max(), 100)
            y_norm = stats.norm.pdf(x_norm, col_data.mean(), col_data.std())
            fig_hist.add_trace(go.Scatter(
                x=x_norm, y=y_norm * len(col_data) * (col_data.max() - col_data.min()) / 50,
                mode='lines', name='Normal Distribution', line=dict(color='red', dash='dash')
            ))
            
            fig_hist.add_vline(x=col_data.mean(), line_dash="dash", line_color="red", 
                             annotation_text="Mean")
            fig_hist.update_layout(showlegend=True)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Enhanced Q-Q plot
            fig_qq = go.Figure()
            qq = stats.probplot(col_data, dist="norm")
            fig_qq.add_trace(go.Scatter(
                x=qq[0][0], y=qq[0][1],
                mode='markers', name='Data Points',
                marker=dict(size=6, opacity=0.6)
            ))
            fig_qq.add_trace(go.Scatter(
                x=qq[0][0], y=qq[0][0] * qq[1][0] + qq[1][1],
                mode='lines', name='Theoretical Normal',
                line=dict(color='red', width=2)
            ))
            fig_qq.update_layout(
                title=f"Q-Q Plot for {selected_col}",
                xaxis_title="Theoretical Quantiles",
                yaxis_title="Sample Quantiles"
            )
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # Descriptive statistics
        st.write("#### Descriptive Statistics")
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Mean", f"{col_data.mean():.4f}")
            st.metric("Standard Deviation", f"{col_data.std():.4f}")
        with stats_col2:
            st.metric("Median", f"{col_data.median():.4f}")
            st.metric("Variance", f"{col_data.var():.4f}")
        with stats_col3:
            skewness = stats.skew(col_data)
            st.metric("Skewness", f"{skewness:.4f}")
            kurtosis = stats.kurtosis(col_data)
            st.metric("Kurtosis", f"{kurtosis:.4f}")
        with stats_col4:
            st.metric("Sample Size", len(col_data))
            st.metric("Range", f"{col_data.max() - col_data.min():.4f}")
        
        # Perform comprehensive normality tests
        if st.button("🚀 Run Comprehensive Normality Analysis", type="primary"):
            with st.spinner("Performing comprehensive normality diagnostics..."):
                results = statistical_tester.perform_normality_tests(selected_col)
                
                if 'error' in results:
                    st.error(f"❌ Error in normality analysis: {results['error']}")
                    return
                
                st.write("#### 📋 Normality Test Results")
                
                # Test results in columns
                test_col1, test_col2, test_col3 = st.columns(3)
                
                with test_col1:
                    shapiro_result = results.get('shapiro', {})
                    if 'error' in shapiro_result:
                        st.error(f"Shapiro-Wilk: {shapiro_result['error']}")
                    else:
                        display_normality_test_result(shapiro_result, "Shapiro-Wilk Test")
                
                with test_col2:
                    dagostino_result = results.get('dagostino', {})
                    if 'error' in dagostino_result:
                        st.error(f"D'Agostino: {dagostino_result['error']}")
                    else:
                        display_normality_test_result(dagostino_result, "D'Agostino Test")
                
                with test_col3:
                    anderson_result = results.get('anderson', {})
                    if 'error' in anderson_result:
                        st.error(f"Anderson-Darling: {anderson_result['error']}")
                    else:
                        display_normality_test_result(anderson_result, "Anderson-Darling Test")
                
                # Overall assessment
                overall = results.get('overall_normality', {})
                if overall:
                    st.write("#### 🎯 Overall Normality Assessment")
                    assessment_col1, assessment_col2 = st.columns(2)
                    
                    with assessment_col1:
                        if overall.get('is_normal', False):
                            st.success("✅ **Overall: Normally Distributed**")
                            st.info("**Recommendation**: Parametric tests (t-tests, ANOVA) are appropriate")
                        else:
                            st.error("❌ **Overall: Not Normally Distributed**")
                            st.info("**Recommendation**: Consider non-parametric tests (Mann-Whitney, Kruskal-Wallis)")
                    
                    with assessment_col2:
                        agreement = "✅ Tests Agree" if overall.get('tests_agree') else "⚠️ Tests Disagree"
                        st.info(f"{agreement} ({overall.get('tests_performed', 0)} tests performed)")
                
                # Outlier analysis
                outliers = results.get('outliers', {})
                if outliers.get('outliers_count', 0) > 0:
                    st.warning(f"⚠️ **Outlier Alert**: {outliers['outliers_count']} outliers detected ({outliers['outlier_percentage']:.1f}% of data)")

def display_normality_test_result(result, test_name):
    """Display individual normality test result"""
    p_value = result.get('p_value')
    statistic = result.get('statistic')
    is_normal = result.get('is_normal')
    
    if p_value is not None:
        st.metric(f"{test_name}", f"p-value: {p_value:.4f}")
        st.write(f"Statistic: {statistic:.4f}")
        
        if is_normal:
            st.success("✅ Normal distribution")
        else:
            st.error("❌ Not normal distribution")
        
        # Practical interpretation
        if p_value < 0.01:
            interpretation = "Strong evidence against normality"
        elif p_value < 0.05:
            interpretation = "Moderate evidence against normality"
        else:
            interpretation = "Insufficient evidence against normality"
        
        st.caption(f"*{interpretation}*")

def render_enhanced_hypothesis_testing(statistical_tester):
    """Enhanced hypothesis testing with assumption checking and guides"""
    
    with st.expander("📚 **Hypothesis Testing Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Hypothesis Tests
        
        **When to Use Each Test:**
        
        **One Sample t-test:**
        - Compare sample mean to known population mean
        - Example: Test if average height differs from national average
        
        **Two Sample t-test:**
        - Compare means between two independent groups
        - Example: Test if drug A works better than drug B
        
        **Paired t-test:**
        - Compare measurements from the same subjects at different times
        - Example: Test before/after treatment effects
        
        **Key Concepts:**
        - **Null Hypothesis (H0)**: No effect/difference
        - **Alternative Hypothesis (H1)**: There is an effect/difference
        - **p-value**: Probability of observing results if H0 is true
        - **Significance Level (α)**: Threshold for rejecting H0 (usually 0.05)
        
        **Interpretation:**
        - **p < α**: Reject H0, evidence for effect
        - **p ≥ α**: Cannot reject H0, no strong evidence for effect
        """)
    
    st.subheader("🔍 Advanced Hypothesis Testing")
    
    df = statistical_tester.df
    numeric_cols = statistical_tester.numeric_cols
    
    if len(numeric_cols) < 2:
        st.info("📊 Need at least 2 numeric columns for hypothesis testing")
        return
    
    test_type = st.selectbox(
        "Select Test Type",
        ["One Sample t-test", "Two Sample t-test", "Paired t-test"],
        key="hyp_test_type"
    )
    
    # Test configuration
    st.write("#### ⚙️ Test Configuration")
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        alpha = st.select_slider("Significance level (α)", 
                               options=[0.001, 0.01, 0.05, 0.10], value=0.05)
        alternative = st.selectbox("Alternative hypothesis",
                                 ["two-sided", "less", "greater"])
    
    with config_col2:
        # Custom alpha option
        custom_alpha = st.checkbox("Use custom alpha value")
        if custom_alpha:
            custom_alpha_val = st.number_input("Custom alpha value", 
                                             min_value=0.001, max_value=0.2, value=0.05, step=0.001)
            alpha = custom_alpha_val
    
    if test_type == "One Sample t-test":
        render_one_sample_ttest(statistical_tester, alpha, alternative)
    
    elif test_type == "Two Sample t-test":
        render_two_sample_ttest(statistical_tester, alpha, alternative)
    
    elif test_type == "Paired t-test":
        render_paired_ttest(statistical_tester, alpha, alternative)

def render_one_sample_ttest(statistical_tester, alpha, alternative):
    """Render one-sample t-test interface"""
    st.write("#### 🎯 One Sample t-test")
    
    numeric_cols = statistical_tester.numeric_cols
    col1, col2 = st.columns(2)
    
    with col1:
        test_col = st.selectbox("Select column", numeric_cols, key="one_sample_col")
        test_mean = st.number_input("Hypothesized population mean", value=0.0, step=0.1)
    
    # Show sample statistics
    if test_col:
        data = statistical_tester.df[test_col].dropna()
        st.info(f"**Sample Statistics**: Mean = {data.mean():.3f}, SD = {data.std():.3f}, N = {len(data)}")
    
    if st.button("Perform One Sample t-test", type="primary"):
        result = statistical_tester.one_sample_ttest(test_col, test_mean, alternative, alpha)
        display_enhanced_ttest_result(result, "One Sample t-test")

def render_two_sample_ttest(statistical_tester, alpha, alternative):
    """Render two-sample t-test interface"""
    st.write("#### 📊 Two Sample t-test")
    
    numeric_cols = statistical_tester.numeric_cols
    col1, col2 = st.columns(2)
    
    with col1:
        col1_select = st.selectbox("First sample column", numeric_cols, key="two_sample_1")
        col2_select = st.selectbox("Second sample column", numeric_cols, key="two_sample_2")
    
    with col2:
        equal_var = st.radio("Variance assumption",
                           ["Auto-detect", "Assume equal", "Assume unequal"])
    
    # Show group statistics
    if col1_select and col2_select:
        data1 = statistical_tester.df[col1_select].dropna()
        data2 = statistical_tester.df[col2_select].dropna()
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Group 1**: Mean = {data1.mean():.3f}, SD = {data1.std():.3f}, N = {len(data1)}")
        with col2:
            st.info(f"**Group 2**: Mean = {data2.mean():.3f}, SD = {data2.std():.3f}, N = {len(data2)}")
    
    if st.button("Perform Two Sample t-test", type="primary"):
        equal_var_option = None if equal_var == "Auto-detect" else (equal_var == "Assume equal")
        result = statistical_tester.two_sample_ttest(
            col1_select, col2_select, alternative, equal_var_option, alpha
        )
        display_enhanced_ttest_result(result, "Two Sample t-test")

def render_paired_ttest(statistical_tester, alpha, alternative):
    """Render paired t-test interface"""
    st.write("#### 🔄 Paired t-test")
    st.info("For paired measurements (e.g., before/after treatment)")
    
    numeric_cols = statistical_tester.numeric_cols
    col1, col2 = st.columns(2)
    
    with col1:
        pre_col = st.selectbox("Pre-treatment column", numeric_cols, key="paired_pre")
        post_col = st.selectbox("Post-treatment column", numeric_cols, key="paired_post")
    
    # Show paired data preview
    paired_data = statistical_tester.df[[pre_col, post_col]].dropna()
    if len(paired_data) > 0:
        st.write(f"**Paired observations**: {len(paired_data)} complete pairs")
        
        # Show pre-post comparison
        fig = go.Figure()
        fig.add_trace(go.Box(y=paired_data[pre_col], name='Pre-treatment', marker_color='lightblue'))
        fig.add_trace(go.Box(y=paired_data[post_col], name='Post-treatment', marker_color='lightcoral'))
        fig.update_layout(title="Pre-treatment vs Post-treatment Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Perform Paired t-test", type="primary"):
        result = statistical_tester.paired_ttest(pre_col, post_col, alternative, alpha)
        display_enhanced_ttest_result(result, "Paired t-test")

def display_enhanced_ttest_result(result, test_name):
    """Display enhanced t-test results with diagnostics"""
    if 'error' in result:
        st.error(f"❌ Error in {test_name}: {result['error']}")
        return
    
    st.write(f"#### 📋 {test_name} Results")
    
    # Main results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("t-statistic", f"{result['t_statistic']:.4f}")
    with col2:
        st.metric("p-value", f"{result['p_value']:.4f}")
    with col3:
        st.metric("Degrees of Freedom", f"{result['df']:.0f}")
    with col4:
        effect_sizes = result.get('effect_sizes', {})
        cohens_d = effect_sizes.get('cohens_d', 'N/A')
        if isinstance(cohens_d, (int, float)):
            st.metric("Effect Size (Cohen's d)", f"{cohens_d:.4f}")
        else:
            st.metric("Effect Size", "N/A")
    
    # Effect size interpretation
    effect_sizes = result.get('effect_sizes', {})
    if 'interpretation' in effect_sizes:
        interpretation = effect_sizes.get('interpretation', 'N/A')
        st.info(f"**Effect Size Interpretation**: {interpretation}")
    
    # Statistical conclusion
    st.write("#### 🎯 Statistical Conclusion")
    if result['reject_null']:
        st.error(f"✅ **Reject null hypothesis** (p < {result['alpha']})")
        st.write("There is statistically significant evidence to support the alternative hypothesis.")
        
        # Practical significance
        if 'effect_sizes' in result and 'cohens_d' in result['effect_sizes']:
            cohens_d = result['effect_sizes']['cohens_d']
            if abs(cohens_d) < 0.2:
                st.info("**Note**: Effect size is very small - may not be practically significant")
            elif abs(cohens_d) < 0.5:
                st.info("**Note**: Effect size is small")
            elif abs(cohens_d) < 0.8:
                st.info("**Note**: Effect size is medium")
            else:
                st.info("**Note**: Effect size is large")
    else:
        st.success(f"❌ **Fail to reject null hypothesis** (p ≥ {result['alpha']})")
        st.write("There is not enough evidence to support the alternative hypothesis.")
    
    # Confidence interval
    if 'confidence_interval' in result:
        ci = result['confidence_interval']
        st.write(f"**{int((1-result['alpha'])*100)}% Confidence Interval:** ({ci[0]:.4f}, {ci[1]:.4f})")
    
    # Assumption checking
    assumptions = result.get('assumptions')
    if assumptions:
        display_assumption_checking(assumptions)

def display_assumption_checking(assumptions):
    """Display test assumption checking results"""
    st.write("#### 🔍 Assumption Checking")
    
    assumption_col1, assumption_col2, assumption_col3 = st.columns(3)
    
    with assumption_col1:
        if assumptions.normality:
            st.success("✅ Normality")
        else:
            st.error("❌ Normality")
    
    with assumption_col2:
        if assumptions.equal_variance:
            st.success("✅ Equal Variance")
        else:
            st.error("❌ Equal Variance")
    
    with assumption_col3:
        if assumptions.sample_size:
            st.success("✅ Sample Size")
        else:
            st.error("❌ Sample Size")
    
    # Warnings
    if assumptions.warnings:
        st.warning("#### ⚠️ Assumption Warnings")
        for warning in assumptions.warnings:
            st.write(f"• {warning}")

def render_enhanced_anova_tests(statistical_tester):
    """Enhanced ANOVA tests with comprehensive diagnostics"""
    
    with st.expander("📚 **ANOVA Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding ANOVA (Analysis of Variance)
        
        **When to Use ANOVA:**
        - Compare means across 3 or more groups
        - Example: Test if test scores differ between multiple teaching methods
        
        **Key Concepts:**
        - **F-statistic**: Ratio of between-group to within-group variance
        - **p-value**: Probability groups have equal means
        - **Post-hoc tests**: Identify which specific groups differ
        
        **Assumptions:**
        - Normality within each group
        - Equal variances between groups
        - Independent observations
        
        **Interpretation:**
        - **p < 0.05**: At least one group mean differs
        - Use Tukey HSD to find which groups differ
        """)
    
    st.subheader("📋 Advanced ANOVA & Group Analysis")
    
    df = statistical_tester.df
    numeric_cols = statistical_tester.numeric_cols
    categorical_cols = statistical_tester.categorical_cols
    
    if not numeric_cols or not categorical_cols:
        st.info("📊 Need both numeric and categorical columns for ANOVA")
        return
    
    st.write("#### 🎯 One-Way ANOVA")
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_col = st.selectbox("Numeric variable", numeric_cols, key="anova_num")
        group_col = st.selectbox("Grouping variable", categorical_cols, key="anova_cat")
    
    with col2:
        alpha = st.select_slider("Significance level", 
                               options=[0.001, 0.01, 0.05, 0.10], value=0.05, key="anova_alpha")
    
    # Show group summaries
    if numeric_col and group_col:
        group_summary = df.groupby(group_col)[numeric_col].agg(['count', 'mean', 'std']).round(3)
        st.write("#### Group Summary Statistics")
        st.dataframe(group_summary, use_container_width=True)
        
        # Visualize groups
        fig = px.box(df, x=group_col, y=numeric_col, title=f"{numeric_col} by {group_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    if st.button("Perform Comprehensive ANOVA", type="primary"):
        with st.spinner("Performing ANOVA with comprehensive diagnostics..."):
            result = statistical_tester.one_way_anova(numeric_col, group_col, alpha)
            
            if 'error' in result:
                st.error(f"❌ Error in ANOVA: {result['error']}")
                return
            
            display_enhanced_anova_result(result, statistical_tester, numeric_col, group_col)

def display_enhanced_anova_result(result, statistical_tester, numeric_col, group_col):
    """Display enhanced ANOVA results"""
    st.write("#### 📋 ANOVA Results")
    
    # Main results
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("F-statistic", f"{result['f_statistic']:.4f}")
    with col2:
        st.metric("p-value", f"{result['p_value']:.4f}")
    with col3:
        st.metric("Groups", result['n_groups'])
    with col4:
        eta_squared = result.get('effect_sizes', {}).get('eta_squared', 'N/A')
        if isinstance(eta_squared, (int, float)):
            st.metric("Effect Size (η²)", f"{eta_squared:.4f}")
        else:
            st.metric("Effect Size", "N/A")
    
    # Statistical conclusion
    st.write("#### 🎯 Statistical Conclusion")
    if result['reject_null']:
        st.error(f"✅ **Reject null hypothesis** (p < {result['alpha']})")
        st.write("There are statistically significant differences between group means.")
    else:
        st.success(f"❌ **Fail to reject null hypothesis** (p ≥ {result['alpha']})")
        st.write("No statistically significant differences between group means.")
    
    # Effect size interpretation
    effect_sizes = result.get('effect_sizes', {})
    if 'interpretation' in effect_sizes:
        st.info(f"**Effect Size Interpretation**: {effect_sizes['interpretation']}")
    
    # Group statistics
    if 'group_stats' in result:
        st.write("#### 📊 Group Statistics")
        group_stats_df = pd.DataFrame(result['group_stats']).T
        st.dataframe(group_stats_df, use_container_width=True)
    
    # Assumption checking
    assumptions = result.get('assumptions')
    if assumptions:
        display_assumption_checking(assumptions)
    
    # Post-hoc analysis if significant
    if result['reject_null'] and result['n_groups'] > 2:
        st.write("#### 🔍 Post-hoc Analysis (Tukey HSD)")
        posthoc_results = statistical_tester.tukey_hsd(numeric_col, group_col)
        if 'error' in posthoc_results.columns and posthoc_results['error'].iloc[0]:
            st.error(f"Tukey HSD Error: {posthoc_results['error'].iloc[0]}")
        else:
            st.dataframe(posthoc_results, use_container_width=True)
            
            # Highlight significant differences
            significant = posthoc_results[posthoc_results['reject'] == True]
            if not significant.empty:
                st.write("**Significant Group Differences:**")
                for _, row in significant.iterrows():
                    st.write(f"- {row['group1']} vs {row['group2']}: p = {row['p_adj']:.4f}")

def render_enhanced_nonparametric_tests(statistical_tester):
    """Enhanced non-parametric tests interface"""
    
    with st.expander("📚 **Non-Parametric Tests Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Non-Parametric Tests
        
        **When to Use Non-Parametric Tests:**
        - Data not normally distributed
        - Small sample sizes
        - Ordinal data or ranks
        - Outliers present
        - Unequal variances
        
        **Common Tests:**
        - **Mann-Whitney U**: Non-parametric t-test for 2 groups
        - **Kruskal-Wallis**: Non-parametric ANOVA for 3+ groups
        - **Wilcoxon Signed-Rank**: Non-parametric paired t-test
        
        **Advantages:**
        - Fewer assumptions
        - Robust to outliers
        - Works with ordinal data
        
        **Disadvantages:**
        - Less statistical power
        - Harder to interpret effect sizes
        """)
    
    st.subheader("⚡ Non-Parametric Tests")
    
    df = statistical_tester.df
    numeric_cols = statistical_tester.numeric_cols
    categorical_cols = statistical_tester.categorical_cols
    
    test_type = st.selectbox(
        "Select Non-Parametric Test",
        ["Mann-Whitney U Test", "Kruskal-Wallis Test", "Wilcoxon Signed-Rank Test"],
        key="nonparam_test"
    )
    
    if test_type == "Mann-Whitney U Test":
        st.write("#### 📊 Mann-Whitney U Test (Two Independent Samples)")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                group1_col = st.selectbox("First sample", numeric_cols, key="mw_1")
                group2_col = st.selectbox("Second sample", numeric_cols, key="mw_2")
            with col2:
                alpha = st.select_slider("Significance level", 
                                       options=[0.01, 0.05, 0.10], value=0.05, key="mw_alpha")
                alternative = st.selectbox("Alternative hypothesis",
                                         ["two-sided", "less", "greater"], key="mw_alt")
            
            if st.button("Perform Mann-Whitney Test", type="primary"):
                result = statistical_tester.mann_whitney_u(
                    group1_col, group2_col, alternative, alpha
                )
                display_nonparametric_result(result, "Mann-Whitney U Test")
    
    elif test_type == "Kruskal-Wallis Test":
        st.write("#### 📋 Kruskal-Wallis Test (Multiple Independent Samples)")
        
        if numeric_cols and categorical_cols:
            numeric_col = st.selectbox("Numeric variable", numeric_cols, key="kw_num")
            group_col = st.selectbox("Grouping variable", categorical_cols, key="kw_cat")
            alpha = st.select_slider("Significance level", 
                                   options=[0.01, 0.05, 0.10], value=0.05, key="kw_alpha")
            
            if st.button("Perform Kruskal-Wallis Test", type="primary"):
                result = statistical_tester.kruskal_wallis(numeric_col, group_col, alpha)
                display_nonparametric_result(result, "Kruskal-Wallis Test")
    
    elif test_type == "Wilcoxon Signed-Rank Test":
        st.write("#### 🔄 Wilcoxon Signed-Rank Test (Paired Samples)")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                pre_col = st.selectbox("Pre-treatment", numeric_cols, key="wilcoxon_pre")
                post_col = st.selectbox("Post-treatment", numeric_cols, key="wilcoxon_post")
            with col2:
                alpha = st.select_slider("Significance level", 
                                       options=[0.01, 0.05, 0.10], value=0.05, key="wilcoxon_alpha")
                alternative = st.selectbox("Alternative hypothesis",
                                         ["two-sided", "less", "greater"], key="wilcoxon_alt")
            
            if st.button("Perform Wilcoxon Test", type="primary"):
                result = statistical_tester.wilcoxon_signed_rank(
                    pre_col, post_col, alternative, alpha
                )
                display_nonparametric_result(result, "Wilcoxon Signed-Rank Test")

def display_nonparametric_result(result, test_name):
    """Display non-parametric test results"""
    if 'error' in result:
        st.error(f"❌ Error in {test_name}: {result['error']}")
        return
    
    st.write(f"#### 📋 {test_name} Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Statistic", f"{result['statistic']:.4f}")
    with col2:
        st.metric("p-value", f"{result['p_value']:.4f}")
    with col3:
        sample_info = result.get('sample_size', result.get('sample_sizes', 'N/A'))
        if isinstance(sample_info, list):
            st.metric("Sample Sizes", f"{sample_info[0]} vs {sample_info[1]}")
        else:
            st.metric("Sample Size", sample_info)
    
    st.write("**Interpretation:**")
    if result['reject_null']:
        st.error(f"✅ Reject null hypothesis (p < {result['alpha']})")
        st.write("There is statistically significant evidence for the alternative hypothesis.")
    else:
        st.success(f"❌ Fail to reject null hypothesis (p ≥ {result['alpha']})")
        st.write("There is not enough evidence for the alternative hypothesis.")

def render_bayesian_analysis(statistical_tester):
    """Render Bayesian statistical analysis"""
    
    with st.expander("📚 **Bayesian Analysis Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Bayesian Statistics
        
        **Key Differences from Classical Statistics:**
        - **Classical**: Probability of data given hypothesis (p-values)
        - **Bayesian**: Probability of hypothesis given data (Bayes factors)
        
        **Bayes Factor Interpretation:**
        - **BF > 100**: Decisive evidence for H1
        - **BF 30-100**: Very strong evidence for H1
        - **BF 10-30**: Strong evidence for H1
        - **BF 3-10**: Moderate evidence for H1
        - **BF 1-3**: Anecdotal evidence for H1
        - **BF 1/3-1**: Anecdotal evidence for H0
        - **BF < 1/3**: Evidence for H0
        
        **Advantages:**
        - More intuitive interpretation
        - Incorporates prior knowledge
        - Provides evidence for both hypotheses
        """)
    
    st.subheader("🔄 Bayesian Statistical Analysis")
    
    st.info("""
    **Note**: Bayesian analysis provides alternative evidence measures using Bayes factors.
    """)
    
    test_type = st.selectbox(
        "Select Bayesian Test",
        ["Bayesian t-test (One Sample)", "Bayesian t-test (Two Sample)", "Bayesian t-test (Paired)"],
        key="bayesian_test"
    )
    
    numeric_cols = statistical_tester.numeric_cols
    
    if test_type == "Bayesian t-test (One Sample)":
        st.write("#### 🎯 Bayesian One Sample t-test")
        
        col1, col2 = st.columns(2)
        with col1:
            test_col = st.selectbox("Select column", numeric_cols, key="bayes_one_sample")
            test_mean = st.number_input("Test mean", value=0.0, step=0.1, key="bayes_mean")
        with col2:
            prior_scale = st.select_slider("Prior scale", 
                                         options=[0.1, 0.5, 0.707, 1.0, 2.0], 
                                         value=0.707,
                                         help="Scale of prior distribution")
        
        if st.button("Run Bayesian t-test", type="primary"):
            with st.spinner("Running Bayesian analysis..."):
                result = statistical_tester.bayesian_ttest(test_col, test_mean=test_mean, prior_scale=prior_scale)
                display_bayesian_result(result, "Bayesian One Sample t-test")
    
    elif "Two Sample" in test_type:
        st.write("#### 📊 Bayesian Two Sample t-test")
        
        col1, col2 = st.columns(2)
        with col1:
            col1_select = st.selectbox("First sample", numeric_cols, key="bayes_two_1")
            col2_select = st.selectbox("Second sample", numeric_cols, key="bayes_two_2")
        with col2:
            prior_scale = st.select_slider("Prior scale", 
                                         options=[0.1, 0.5, 0.707, 1.0, 2.0], 
                                         value=0.707,
                                         key="bayes_two_prior")
        
        if st.button("Run Bayesian Two Sample t-test", type="primary"):
            with st.spinner("Running Bayesian analysis..."):
                result = statistical_tester.bayesian_ttest(col1_select, col2_select, prior_scale=prior_scale)
                display_bayesian_result(result, "Bayesian Two Sample t-test")
    
    elif "Paired" in test_type:
        st.write("#### 🔄 Bayesian Paired t-test")
        
        col1, col2 = st.columns(2)
        with col1:
            pre_col = st.selectbox("Pre-treatment", numeric_cols, key="bayes_pre")
            post_col = st.selectbox("Post-treatment", numeric_cols, key="bayes_post")
        with col2:
            prior_scale = st.select_slider("Prior scale", 
                                         options=[0.1, 0.5, 0.707, 1.0, 2.0], 
                                         value=0.707,
                                         key="bayes_paired_prior")
        
        if st.button("Run Bayesian Paired t-test", type="primary"):
            with st.spinner("Running Bayesian analysis..."):
                result = statistical_tester.bayesian_ttest(pre_col, post_col, paired=True, prior_scale=prior_scale)
                display_bayesian_result(result, "Bayesian Paired t-test")

def display_bayesian_result(result, test_name):
    """Display Bayesian test results"""
    if 'error' in result:
        st.error(f"❌ Error in {test_name}: {result['error']}")
        return
    
    st.write(f"#### 📋 {test_name} Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Bayes Factor", f"{result['bayes_factor']:.4f}")
    with col2:
        st.metric("Evidence Strength", result['evidence_strength'])
    with col3:
        st.metric("Test Type", "Paired" if result.get('paired') else "Independent")
    
    st.write("#### 🎯 Bayesian Interpretation")
    st.info(f"**{result['interpretation']}**")
    
    # Bayes factor guide
    st.write("#### 📊 Bayes Factor Guide")
    bf_guide = {
        "> 100": "Decisive evidence for H1",
        "30 - 100": "Very strong evidence for H1",
        "10 - 30": "Strong evidence for H1",
        "3 - 10": "Moderate evidence for H1",
        "1 - 3": "Anecdotal evidence for H1",
        "1/3 - 1": "Anecdotal evidence for H0",
        "< 1/3": "Evidence for H0"
    }
    
    for range_str, interpretation in bf_guide.items():
        st.write(f"- **BF {range_str}**: {interpretation}")

def render_time_series_analysis(statistical_tester):
    """Render time series analysis"""
    
    with st.expander("📚 **Time Series Analysis Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Time Series Analysis
        
        **Key Concepts:**
        - **Trend**: Long-term increase or decrease
        - **Seasonality**: Regular periodic fluctuations
        - **Stationarity**: Statistical properties constant over time
        - **Autocorrelation**: Correlation with past values
        
        **Stationarity Test (ADF):**
        - **p < 0.05**: Series is stationary
        - **p ≥ 0.05**: Series may have trend/seasonality
        
        **Decomposition:**
        - **Trend**: Overall direction
        - **Seasonal**: Regular patterns
        - **Residual**: Random noise
        
        **Applications:**
        - Sales forecasting
        - Stock price analysis
        - Weather patterns
        - Economic indicators
        """)
    
    st.subheader("⏰ Time Series Analysis")
    
    st.info("""
    **Note**: Time series analysis requires a date/time column or will use sequential indexing.
    """)
    
    numeric_cols = statistical_tester.numeric_cols
    all_cols = statistical_tester.df.columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        ts_column = st.selectbox("Time series column", numeric_cols, key="ts_col")
    with col2:
        date_column = st.selectbox("Date/Time column (optional)", [None] + all_cols, key="date_col")
    
    frequency = st.selectbox("Data frequency", 
                           ["D (Daily)", "W (Weekly)", "M (Monthly)", "Q (Quarterly)", "Y (Yearly)"],
                           help="Approximate frequency of your data")
    
    if st.button("Analyze Time Series", type="primary"):
        with st.spinner("Performing time series analysis..."):
            result = statistical_tester.time_series_analysis(
                ts_column, date_column, frequency.split(" ")[0]
            )
            
            if 'error' in result:
                st.error(f"❌ Error in time series analysis: {result['error']}")
                return
            
            display_time_series_results(result, ts_column)

def display_time_series_results(result, column_name):
    """Display time series analysis results"""
    st.write("#### 📋 Time Series Analysis Results")
    
    # Stationarity test
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Stationary", "Yes" if result['stationary'] else "No")
    with col2:
        st.metric("ADF Statistic", f"{result['adf_statistic']:.4f}")
    with col3:
        st.metric("ADF p-value", f"{result['adf_pvalue']:.4f}")
    
    st.write("#### 🎯 Stationarity Interpretation")
    if result['stationary']:
        st.success("✅ **Series is stationary** - statistical properties are constant over time")
        st.write("Suitable for many time series models without differencing.")
    else:
        st.warning("⚠️ **Series may not be stationary** - consider differencing or transformation")
        st.write("May contain trend, seasonality, or other time-dependent patterns.")
    
    # ACF and PACF plots
    st.write("#### 📊 Autocorrelation Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(y=result['acf'][1:], name='ACF'))
        fig_acf.update_layout(title="Autocorrelation Function (ACF)", 
                            xaxis_title="Lag", yaxis_title="ACF")
        st.plotly_chart(fig_acf, use_container_width=True)
    
    with col2:
        fig_pacf = go.Figure()
        fig_pacf.add_trace(go.Bar(y=result['pacf'][1:], name='PACF'))
        fig_pacf.update_layout(title="Partial Autocorrelation Function (PACF)",
                             xaxis_title="Lag", yaxis_title="PACF")
        st.plotly_chart(fig_pacf, use_container_width=True)

def render_survival_analysis(statistical_tester):
    """Render survival analysis interface"""
    
    with st.expander("📚 **Survival Analysis Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Survival Analysis
        
        **When to Use:**
        - Time-to-event data
        - Customer churn analysis
        - Medical studies (time to recovery/death)
        - Equipment failure times
        - Marketing conversion times
        
        **Key Concepts:**
        - **Survival Function**: Probability of surviving past time t
        - **Hazard Function**: Instantaneous risk of event
        - **Censoring**: Incomplete observation of event time
        
        **Tests:**
        - **Kaplan-Meier**: Non-parametric survival curves
        - **Cox Regression**: Multivariate survival analysis
        
        **Interpretation:**
        - **Hazard Ratio > 1**: Increased risk
        - **Hazard Ratio < 1**: Decreased risk
        - **Hazard Ratio = 1**: No effect
        """)
    
    st.subheader("📉 Survival Analysis")
    
    st.info("""
    **Note**: Survival analysis requires lifelines package.
    Install with: `pip install lifelines`
    """)
    
    numeric_cols = statistical_tester.numeric_cols
    categorical_cols = statistical_tester.categorical_cols
    
    col1, col2, col3 = st.columns(3)
    with col1:
        time_col = st.selectbox("Time to event column", numeric_cols, key="survival_time")
    with col2:
        event_col = st.selectbox("Event occurrence column", numeric_cols, key="survival_event")
    with col3:
        group_col = st.selectbox("Grouping column (optional)", [None] + categorical_cols, key="survival_group")
    
    if st.button("Perform Survival Analysis", type="primary"):
        with st.spinner("Running survival analysis..."):
            result = statistical_tester.survival_analysis(time_col, event_col, group_col)
            
            if 'error' in result:
                st.error(f"❌ Error in survival analysis: {result['error']}")
                return
            
            display_survival_results(result, time_col, event_col, group_col)

def display_survival_results(result, time_col, event_col, group_col):
    """Display survival analysis results"""
    st.write("#### 📋 Survival Analysis Results")
    
    if 'median_survival' in result and result['median_survival'] is not None:
        st.metric("Median Survival Time", f"{result['median_survival']:.2f}")
    
    # Kaplan-Meier curve data available
    if 'km_curve' in result and hasattr(result['km_curve'], 'shape'):
        st.success(f"✅ Kaplan-Meier analysis completed with {result['km_curve'].shape[0]} time points")
    
    # Cox regression results
    if result.get('cox_results'):
        st.write("#### 🔍 Cox Proportional Hazards Model")
        cox_summary = result['cox_results']['summary']
        st.write("**Hazard Ratios:**")
        for var, hr in result['cox_results']['hazard_ratios'].items():
            interpretation = "Increased risk" if hr > 1 else "Decreased risk" if hr < 1 else "No effect"
            st.write(f"- **{var}**: HR = {hr:.3f} ({interpretation})")
    else:
        st.info("No Cox regression results available (grouping variable not provided or insufficient data)")

def render_multivariate_analysis(statistical_tester):
    """Render multivariate analysis interface"""
    
    with st.expander("📚 **Multivariate Analysis Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Multivariate Analysis
        
        **MANOVA (Multivariate ANOVA):**
        - Tests mean differences across multiple dependent variables
        - Example: Test if teaching methods affect both math and reading scores
        
        **Factor Analysis:**
        - Identifies underlying latent variables (factors)
        - Reduces dimensionality
        - Example: Identify personality traits from survey questions
        
        **When to Use MANOVA:**
        - Multiple correlated dependent variables
        - Want to control Type I error rate
        - Theoretical reason to analyze variables together
        
        **When to Use Factor Analysis:**
        - Many correlated variables
        - Want to reduce dimensionality
        - Looking for underlying constructs
        """)
    
    st.subheader("🌐 Multivariate Analysis")
    
    analysis_type = st.selectbox(
        "Select Multivariate Analysis",
        ["MANOVA", "Factor Analysis"],
        key="multivariate_type"
    )
    
    numeric_cols = statistical_tester.numeric_cols
    categorical_cols = statistical_tester.categorical_cols
    
    if analysis_type == "MANOVA":
        st.write("#### 📊 MANOVA (Multivariate Analysis of Variance)")
        
        col1, col2 = st.columns(2)
        with col1:
            dependent_vars = st.multiselect("Select dependent variables", numeric_cols, key="manova_dep")
        with col2:
            group_var = st.selectbox("Select grouping variable", categorical_cols, key="manova_group")
        
        if len(dependent_vars) < 2:
            st.warning("MANOVA requires at least 2 dependent variables")
            return
        
        if st.button("Run MANOVA", type="primary"):
            with st.spinner("Performing MANOVA..."):
                result = statistical_tester.manova_analysis(dependent_vars, group_var)
                
                if 'error' in result:
                    st.error(f"❌ Error in MANOVA: {result['error']}")
                    return
                
                st.write("#### 📋 MANOVA Results")
                st.success(f"✅ MANOVA completed for {len(dependent_vars)} dependent variables")
                st.write("**Analysis Summary:**")
                st.text(result['manova_results'])
    
    else:  # Factor Analysis
        st.write("#### 🔍 Exploratory Factor Analysis")
        
        selected_vars = st.multiselect("Select variables for factor analysis", numeric_cols, key="factor_vars")
        n_factors = st.slider("Number of factors to extract", 1, min(5, len(selected_vars)), 2,
                            help="Number of underlying factors to identify")
        
        if len(selected_vars) < 3:
            st.warning("Factor analysis requires at least 3 variables")
            return
        
        if st.button("Run Factor Analysis", type="primary"):
            with st.spinner("Performing factor analysis..."):
                result = statistical_tester.factor_analysis(selected_vars, n_factors)
                
                if 'error' in result:
                    st.error(f"❌ Error in factor analysis: {result['error']}")
                    return
                
                display_factor_analysis_results(result, selected_vars)

def display_factor_analysis_results(result, variables):
    """Display factor analysis results"""
    st.write("#### 📋 Factor Analysis Results")
    
    st.metric("Number of Factors", result['n_factors'])
    st.metric("Number of Variables", len(variables))
    
    # Variance explained
    variance_info = result['variance_explained']
    st.write("#### 📊 Variance Explained")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SS Loadings", f"{variance_info[0].sum():.3f}")
    with col2:
        st.metric("Proportion Var", f"{variance_info[1].sum():.3f}")
    with col3:
        st.metric("Cumulative Var", f"{variance_info[2].sum():.3f}")
    
    st.info(f"**Total variance explained**: {variance_info[1].sum():.1%}")

def render_power_analysis(statistical_tester):
    """Render statistical power analysis"""
    
    with st.expander("📚 **Power Analysis Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding Statistical Power
        
        **What is Power?**
        - Probability of detecting an effect if it exists
        - Power = 1 - β (where β is Type II error rate)
        
        **Key Elements:**
        - **Effect Size**: Magnitude of the effect (Cohen's d, η², etc.)
        - **Sample Size**: Number of observations
        - **Significance Level (α)**: Type I error rate (usually 0.05)
        - **Power (1-β)**: Desired probability to detect effect (usually 0.8)
        
        **Common Guidelines:**
        - **Small effect**: d = 0.2, η² = 0.01
        - **Medium effect**: d = 0.5, η² = 0.06  
        - **Large effect**: d = 0.8, η² = 0.14
        
        **Use Cases:**
        - Plan sample size for new studies
        - Check if existing study had sufficient power
        - Interpret non-significant results
        """)
    
    st.subheader("🎯 Statistical Power Analysis")
    
    st.write("#### 📊 Sample Size and Power Calculator")
    
    test_type = st.selectbox(
        "Select Test Type for Power Analysis",
        ["ttest", "anova", "chi2"],
        key="power_analysis"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        effect_size = st.select_slider(
            "Effect Size",
            options=[0.1, 0.2, 0.3, 0.5, 0.8],
            value=0.5,
            help="Small=0.2, Medium=0.5, Large=0.8"
        )
    
    with col2:
        alpha = st.select_slider(
            "Significance Level (α)",
            options=[0.001, 0.01, 0.05, 0.10],
            value=0.05
        )
    
    with col3:
        power = st.select_slider(
            "Desired Power (1-β)",
            options=[0.7, 0.8, 0.9, 0.95],
            value=0.8
        )
    
    if st.button("Calculate Required Sample Size", type="primary"):
        result = statistical_tester.power_analysis(test_type, effect_size, alpha, power)
        
        if 'error' in result:
            st.error(f"❌ Power analysis error: {result['error']}")
        else:
            st.success("#### 📋 Power Analysis Results")
            
            if test_type == 'ttest':
                st.metric("Required Sample Size", result['required_sample_size'])
            elif test_type == 'anova':
                st.metric("Required Sample Size per Group", result['required_sample_size_per_group'])
            else:
                st.metric("Required Sample Size", result['required_sample_size'])
            
            effect_size_names = {
                0.1: "very small",
                0.2: "small", 
                0.3: "small to medium",
                0.5: "medium",
                0.8: "large"
            }
            
            st.info(f"""
            **Interpretation**:
            - To detect a **{effect_size_names[effect_size]}** effect size
            - With **{power*100}%** power (1-β)
            - At **α = {alpha}** significance level
            - You need **{result['required_sample_size'] if test_type == 'ttest' else result['required_sample_size_per_group']}** observations
            """)
            
            # Current data comparison
            current_n = len(statistical_tester.df)
            required_n = result['required_sample_size'] if test_type == 'ttest' else result['required_sample_size_per_group']
            
            if current_n >= required_n:
                st.success(f"✅ Your current sample size ({current_n}) is sufficient")
            else:
                st.warning(f"⚠️ Your current sample size ({current_n}) may be too small. Consider collecting more data.")

def render_comprehensive_summary(statistical_tester):
    """Render comprehensive statistical summary"""
    
    with st.expander("📚 **Comprehensive Summary Guide - Click to Expand**", expanded=False):
        st.markdown("""
        ### Understanding the Comprehensive Summary
        
        **What You'll Get:**
        - Dataset overview and data quality assessment
        - Normality assessment for key variables
        - Correlation analysis insights
        - Statistical power evaluation
        - Actionable recommendations
        
        **How to Use This Summary:**
        1. **Review data quality** - address any issues first
        2. **Check normality** - guides test selection
        3. **Review correlations** - identify relationships
        4. **Follow recommendations** - implement suggested actions
        5. **Use for reporting** - share with stakeholders
        
        **Key Metrics to Watch:**
        - Missing data > 20%: May need imputation
        - Sample size < 30: Consider non-parametric tests
        - Strong correlations: May indicate multicollinearity
        - Non-normal data: Use appropriate tests
        """)
    
    st.subheader("📋 Comprehensive Statistical Summary")
    
    if st.button("🚀 Generate Comprehensive Analysis Report", type="primary"):
        with st.spinner("Generating comprehensive statistical analysis report..."):
            summary = statistical_tester.generate_comprehensive_summary()
            
            # Dataset Overview
            st.write("## 📊 Dataset Overview")
            overview = summary['dataset_overview']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Observations", overview['total_observations'])
            with col2:
                st.metric("Numeric Variables", overview['numeric_variables'])
            with col3:
                st.metric("Categorical Variables", overview['categorical_variables'])
            with col4:
                st.metric("Missing Data", f"{overview['missing_data_percentage']:.1f}%")
            
            # Data Quality Assessment
            st.write("## 🔍 Data Quality Assessment")
            quality = summary['data_quality']
            st.write(f"**Overall Data Completeness**: {quality['completeness_score']:.1f}%")
            
            if quality['high_missing_columns']:
                st.warning(f"⚠️ **High Missing Data Columns**: {', '.join(quality['high_missing_columns'][:3])}")
            else:
                st.success("✅ No columns with high missing data (>20%)")
            
            # Normality Assessment
            st.write("## 📈 Normality Assessment")
            normality = summary['normality_assessment']
            if normality:
                normality_df = pd.DataFrame(normality).T
                st.dataframe(normality_df, use_container_width=True)
            else:
                st.info("No normality assessment available")
            
            # Advanced Insights
            st.write("## 🎯 Advanced Insights")
            insights = summary['advanced_insights']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Parametric-ready Columns:**")
                if insights['parametric_ready_columns']:
                    for col in insights['parametric_ready_columns']:
                        st.success(f"✅ {col}")
                else:
                    st.info("No columns passed normality tests")
            
            with col2:
                st.write("**Non-Parametric Columns:**")
                if insights['non_parametric_columns']:
                    for col in insights['non_parametric_columns']:
                        st.warning(f"⚠️ {col}")
                else:
                    st.info("All columns appear normally distributed")
            
            # Recommendations
            st.write("## 💡 Recommendations")
            recommendations = summary['recommendations']
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.success("✅ No major issues detected. Data appears ready for analysis!")
            
            # Statistical Power
            st.write("## ⚡ Statistical Power Assessment")
            power_info = summary['statistical_power']
            st.info(f"""
            **Current Sample Size**: {power_info['sample_size']} observations
            **Power Status**: {power_info['power_status']}
            **Recommended Minimum**: {power_info['recommended_min_sample']} observations
            """)
            
            # Export options
            st.write("## 📤 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export Summary as JSON"):
                    try:
                        export_data = statistical_tester.export_analysis_history('json')
                        st.download_button(
                            label="Download JSON Report",
                            data=export_data,
                            file_name=f"statistical_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with col2:
                if st.button("Export Summary as CSV"):
                    try:
                        export_data = statistical_tester.export_analysis_history('csv')
                        st.download_button(
                            label="Download CSV Report",
                            data=export_data,
                            file_name=f"statistical_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")

# Make sure to add this function to handle the expanded functionality
def check_advanced_dependencies():
    """Check if advanced statistical dependencies are available"""
    try:
        from statsmodels.stats.power import TTestPower
        from lifelines import KaplanMeierFitter
        return True
    except ImportError:
        return False

# Initialize advanced features availability
ADVANCED_FEATURES_AVAILABLE = check_advanced_dependencies()