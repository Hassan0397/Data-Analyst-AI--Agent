import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import tempfile
import os
import warnings
import zipfile
import json
from scipy import stats

warnings.filterwarnings('ignore')

def render_data_overview():
    """Render enhanced data overview with comprehensive analysis"""
    
    st.header("📋 Data Overview & Quality Analysis")
    
    if not st.session_state.get('processed_data'):
        st.warning("No data available. Please upload files first.")
        return
    
    # File selector
    file_names = list(st.session_state.processed_data.keys())
    selected_file = st.selectbox(
        "Select File to Analyze",
        file_names,
        index=file_names.index(st.session_state.current_file) if st.session_state.current_file in file_names else 0
    )
    
    st.session_state.current_file = selected_file
    
    data_info = st.session_state.processed_data[selected_file]
    df = data_info['dataframe'].copy()
    
    # Auto-detect and convert datetime columns
    df, datetime_conversions = auto_detect_and_convert_datetime(df)
    
    # Show datetime conversion results
    if datetime_conversions:
        st.success("🕒 Auto-detected and converted datetime columns:")
        for col, original_type, new_type in datetime_conversions:
            st.write(f"  • **{col}**: {original_type} → {new_type}")
    
    # Update the dataframe in session state
    st.session_state.processed_data[selected_file]['dataframe'] = df
    
    # Enhanced Data Quality Dashboard
    st.subheader("📊 Data Quality Dashboard")
    
    # Quality metrics in columns
    q1, q2, q3, q4 = st.columns(4)
    
    with q1:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with q2:
        duplicates = df.duplicated().sum()
        st.metric("Duplicate Rows", f"{duplicates:,}")
    
    with q3:
        constant_cols = len([col for col in df.columns if df[col].nunique() == 1])
        st.metric("Constant Columns", f"{constant_cols}")
    
    with q4:
        optimization_potential = suggest_memory_optimization(df)
        st.metric("Memory Optimization", f"{optimization_potential:.1f}%")
    
    # Basic information cards
    st.markdown("---")
    st.subheader("📈 Dataset Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    with col3:
        memory_mb = df.memory_usage(deep=True).sum() / (1024**2) if len(df) > 0 else 0
        st.metric("Memory Usage", f"{memory_mb:.2f} MB")
    with col4:
        missing_values = df.isnull().sum().sum()
        st.metric("Missing Values", f"{missing_values:,}")
    
    st.markdown("---")
    
    # Automated Insights
    st.subheader("🤖 Automated Insights")
    insights = get_enhanced_data_overview_insights(df)
    
    for insight in insights:
        if "Warning:" in insight or "🚨" in insight:
            st.warning(insight)
        elif "✅" in insight or "Success" in insight:
            st.success(insight)
        else:
            st.info(insight)
    
    st.markdown("---")
    
    # Data preview
    st.subheader("🔍 Data Preview")
    
    preview_tab1, preview_tab2, preview_tab3, preview_tab4 = st.tabs([
        "First 10 Rows", "Last 10 Rows", "Random Sample", "Data Types"
    ])
    
    with preview_tab1:
        display_dataframe_safe(df.head(10), "First 10 rows")
    
    with preview_tab2:
        display_dataframe_safe(df.tail(10), "Last 10 rows")
    
    with preview_tab3:
        sample_size = min(10, len(df))
        display_dataframe_safe(df.sample(sample_size) if len(df) > 0 else df, "Random sample")
    
    with preview_tab4:
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        display_dataframe_safe(dtype_info, "Data types overview")
    
    # Enhanced Data Information
    st.markdown("---")
    st.subheader("📊 Advanced Data Analysis")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.write("**📋 Column Information**")
        column_info = create_enhanced_column_info_table(df)
        display_dataframe_safe(column_info, "Column information")
        
        # Data Type Optimization Suggestions
        st.write("**💾 Memory Optimization Suggestions**")
        optimization_suggestions = suggest_data_type_optimization(df)
        if optimization_suggestions:
            for suggestion in optimization_suggestions[:3]:  # Show top 3
                st.info(
                    f"**{suggestion['column']}**: {suggestion['current_type']} → "
                    f"{suggestion['suggested_type']} (Save {suggestion['memory_saving_mb']:.2f} MB)"
                )
        else:
            st.success("Data types are already optimized!")
    
    with info_col2:
        st.write("**📊 Data Types Distribution**")
        dtype_counts = create_dtype_counts(df)
        if not dtype_counts.empty:
            fig = px.pie(
                values=dtype_counts['count'], 
                names=dtype_counts.index,
                title="Data Types Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for chart")
        
        st.write("**🔍 Missing Values Overview**")
        missing_data = create_enhanced_missing_data_summary(df)
        if not missing_data.empty:
            fig = px.bar(
                missing_data, 
                x=missing_data.index,
                y='missing_count',
                title="Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")
    
    # Enhanced Statistical summary
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats_tab1, stats_tab2 = st.tabs(["Basic Statistics", "Advanced Analysis"])
        
        with stats_tab1:
            stats_df = numeric_df.describe()
            display_dataframe_safe(stats_df, "Statistical summary")
            
            # Correlation matrix
            if len(numeric_df.columns) > 1:
                st.write("**📊 Correlation Matrix**")
                corr_matrix = numeric_df.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with stats_tab2:
            # Distribution analysis
            st.write("**📈 Distribution Analysis**")
            selected_num_col = st.selectbox(
                "Select numeric column for distribution:",
                numeric_df.columns,
                key="dist_col"
            )
            if selected_num_col:
                fig = px.histogram(
                    numeric_df, 
                    x=selected_num_col,
                    title=f"Distribution of {selected_num_col}",
                    marginal="box"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for statistical summary.")
    
    # Enhanced Categorical data summary
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        st.markdown("---")
        st.subheader("📊 Categorical Data Analysis")
        
        for col in categorical_df.columns:
            with st.expander(f"**{col}** - {df[col].nunique()} unique values"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Value counts
                    value_counts = df[col].value_counts().head(10)
                    st.write("**Top 10 Values:**")
                    for value, count in value_counts.items():
                        st.write(f"- {value}: {count:,} ({count/len(df)*100:.1f}%)")
                
                with col2:
                    # Visualization
                    top_values = df[col].value_counts().head(8)
                    if not top_values.empty:
                        fig = px.bar(
                            x=top_values.index.astype(str),
                            y=top_values.values,
                            title=f"Top Values in {col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # Datetime Analysis
    datetime_df = df.select_dtypes(include=['datetime64'])
    if not datetime_df.empty:
        st.markdown("---")
        st.subheader("🕒 Datetime Analysis")
        
        for col in datetime_df.columns:
            with st.expander(f"**{col}** - Datetime Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Date Range:**")
                    st.write(f"- Start: {df[col].min()}")
                    st.write(f"- End: {df[col].max()}")
                    st.write(f"- Duration: {df[col].max() - df[col].min()}")
                    
                    # Date distribution
                    date_counts = df[col].dt.date.value_counts().head(10)
                    st.write("**Top 10 Dates:**")
                    for date, count in date_counts.items():
                        st.write(f"- {date}: {count:,}")
                
                with col2:
                    # Time series plot
                    try:
                        daily_counts = df[col].dt.date.value_counts().sort_index()
                        fig = px.line(
                            x=daily_counts.index,
                            y=daily_counts.values,
                            title=f"Daily Frequency - {col}",
                            labels={'x': 'Date', 'y': 'Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.info("Could not generate time series plot")
    
    # Enhanced Quick Analysis Section
    st.markdown("---")
    st.subheader("🚀 Advanced Quick Analysis")
    generate_enhanced_quick_analysis(df)
    
    # Data Quality Issues Section
    st.markdown("---")
    st.subheader("🚨 Data Quality Issues")
    
    quality_issues = identify_data_quality_issues(df)
    if quality_issues:
        for issue in quality_issues:
            st.error(issue)
    else:
        st.success("No major data quality issues detected!")
    
    # ENHANCED Export Functionality
    st.markdown("---")
    st.subheader("📤 Professional Export Options")
    
    # Export configuration
    st.write("### ⚙️ Export Configuration")
    export_config_col1, export_config_col2 = st.columns(2)
    
    with export_config_col1:
        include_charts = st.checkbox("Include Interactive Charts", value=True)
        include_statistics = st.checkbox("Include Detailed Statistics", value=True)
        include_samples = st.checkbox("Include Data Samples", value=True)
        
    with export_config_col2:
        report_format = st.selectbox(
            "Report Format",
            ["Comprehensive HTML", "Executive Summary", "Technical Deep Dive"]
        )
        export_name = st.text_input(
            "Report Name",
            value=f"data_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Enhanced export buttons with progress tracking
    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
    
    with export_col1:
        if st.button("📊 Export Data Profile", use_container_width=True, key="export_profile"):
            with st.spinner("Generating comprehensive data profile..."):
                try:
                    profile_data = generate_enhanced_data_profile(df, include_charts, include_statistics)
                    st.download_button(
                        label="📥 Download Data Profile",
                        data=profile_data,
                        file_name=f"{export_name}_profile.html",
                        mime="text/html",
                        key="download_profile"
                    )
                    st.success("✅ Data profile generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating profile: {str(e)}")
    
    with export_col2:
        if st.button("🔄 Data Quality Report", use_container_width=True, key="export_quality"):
            with st.spinner("Generating professional quality report..."):
                try:
                    quality_report = generate_professional_quality_report(df, report_format)
                    st.download_button(
                        label="📥 Download Quality Report",
                        data=quality_report,
                        file_name=f"{export_name}_quality.html",
                        mime="text/html",
                        key="download_quality"
                    )
                    st.success("✅ Quality report generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating quality report: {str(e)}")
    
    with export_col3:
        if st.button("📈 Analysis Dashboard", use_container_width=True, key="export_dashboard"):
            with st.spinner("Creating comprehensive analysis dashboard..."):
                try:
                    dashboard_report = generate_analysis_dashboard(df, include_charts, include_samples)
                    st.download_button(
                        label="📥 Download Dashboard",
                        data=dashboard_report,
                        file_name=f"{export_name}_dashboard.html",
                        mime="text/html",
                        key="download_dashboard"
                    )
                    st.success("✅ Analysis dashboard generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating dashboard: {str(e)}")
    
    with export_col4:
        if st.button("📁 Export Raw Data", use_container_width=True, key="export_raw"):
            with st.spinner("Preparing data export..."):
                try:
                    csv_data = export_enhanced_csv(df)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"{export_name}_cleaned.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                    st.success("✅ Data exported successfully!")
                except Exception as e:
                    st.error(f"❌ Error exporting data: {str(e)}")
    
    # Batch export option
    st.markdown("---")
    st.write("### 🚀 Batch Export Operations")
    
    batch_col1, batch_col2 = st.columns(2)
    
    with batch_col1:
        if st.button("🎁 Export Complete Analysis Suite", use_container_width=True, key="export_suite"):
            with st.spinner("Generating complete analysis suite... This may take a moment."):
                try:
                    zip_buffer = generate_complete_analysis_suite(df, export_name)
                    st.download_button(
                        label="📦 Download Complete Suite (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"{export_name}_complete_suite.zip",
                        mime="application/zip",
                        key="download_suite"
                    )
                    st.success("✅ Complete analysis suite generated!")
                except Exception as e:
                    st.error(f"❌ Error generating suite: {str(e)}")
    
    with batch_col2:
        if st.button("📋 Generate Executive Summary", use_container_width=True, key="export_exec_summary"):
            with st.spinner("Creating executive summary..."):
                try:
                    exec_summary = generate_executive_summary(df)
                    st.download_button(
                        label="📄 Download Executive Summary",
                        data=exec_summary,
                        file_name=f"{export_name}_executive_summary.html",
                        mime="text/html",
                        key="download_exec_summary"
                    )
                    st.success("✅ Executive summary generated!")
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")

# ===============================
# ENHANCED QUICK ANALYSIS FUNCTIONS (DEBUGGED)
# ===============================

def generate_enhanced_quick_analysis(df):
    """Generate advanced interactive quick analysis with real-time updates"""
    
    st.write("### 🔍 Interactive Data Explorer")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Scatter Plot", "Histogram", "Box Plot", "Correlation Analysis", "Time Series", "Categorical Analysis"],
        key="analysis_type_main"
    )
    
    if analysis_type == "Scatter Plot":
        scatter_analysis(df)
    elif analysis_type == "Histogram":
        histogram_analysis(df)
    elif analysis_type == "Box Plot":
        box_plot_analysis(df)
    elif analysis_type == "Correlation Analysis":
        correlation_analysis(df)
    elif analysis_type == "Time Series":
        time_series_analysis(df)
    elif analysis_type == "Categorical Analysis":
        categorical_analysis(df)

def scatter_analysis(df):
    """Interactive scatter plot analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_axis = st.selectbox("X-axis", numeric_cols, key="scatter_x")
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        with col3:
            color_options = ["None"] + [col for col in df.columns if df[col].nunique() <= 20]
            color_by = st.selectbox("Color by", color_options, key="scatter_color")
        
        # Additional filters
        st.write("**Advanced Filters**")
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            size_options = ["None"] + numeric_cols
            size_by = st.selectbox("Size by", size_options, key="scatter_size")
        with filter_col2:
            facet_options = ["None"] + [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() <= 5]
            facet_by = st.selectbox("Facet by", facet_options, key="scatter_facet")
        
        # Generate plot
        if x_axis and y_axis:  # This condition is fine since x_axis and y_axis are strings
            try:
                plot_data = df.copy()
                
                # Handle color mapping - convert "None" to None
                color_col = None if color_by == "None" else color_by
                size_col = None if size_by == "None" else size_by
                facet_col = None if facet_by == "None" else facet_by
                
                # Create the scatter plot
                fig = px.scatter(
                    plot_data, 
                    x=x_axis, 
                    y=y_axis,
                    color=color_col,
                    size=size_col,
                    facet_col=facet_col,
                    title=f"{y_axis} vs {x_axis}",
                    hover_data=plot_data.columns.tolist()[:5] if len(plot_data.columns) >= 5 else plot_data.columns.tolist()
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical insights - FIXED: Use proper boolean condition
                valid_data_mask = df[[x_axis, y_axis]].notna().all(axis=1)
                valid_data = df.loc[valid_data_mask, [x_axis, y_axis]]
                
                if len(valid_data) > 1:  # This condition is safe
                    correlation = valid_data[x_axis].corr(valid_data[y_axis])
                    strength = "Strong" if abs(correlation) > 0.7 else "Moderate" if abs(correlation) > 0.3 else "Weak"
                    st.info(f"**Correlation Coefficient**: {correlation:.3f} ({strength} relationship)")
                else:
                    st.warning("Not enough valid data to calculate correlation")
                
            except Exception as e:
                st.error(f"Error generating scatter plot: {str(e)}")
    else:
        st.warning("Need at least 2 numeric columns for scatter plot analysis")

def histogram_analysis(df):
    """Interactive histogram analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox("Select Column", numeric_cols, key="hist_col")
        with col2:
            bins = st.slider("Number of Bins", 5, 100, 30, key="hist_bins")
        
        # Additional options
        show_distribution = st.checkbox("Show Distribution Curve", value=True, key="hist_dist")
        color_options = ["None"] + [col for col in df.select_dtypes(include=['object', 'category']).columns if df[col].nunique() <= 10]
        color_by = st.selectbox("Color by Group", color_options, key="hist_color")
        
        if column:
            try:
                plot_data = df[[column]].dropna()
                if color_by != "None":
                    plot_data[color_by] = df[color_by]
                
                fig = px.histogram(
                    plot_data, 
                    x=column,
                    nbins=bins,
                    color=color_by if color_by != "None" else None,
                    marginal="box",
                    title=f"Distribution of {column}",
                    opacity=0.7
                )
                
                if show_distribution and color_by == "None":
                    # Add normal distribution curve
                    try:
                        x_range = np.linspace(plot_data[column].min(), plot_data[column].max(), 100)
                        pdf = stats.norm.pdf(x_range, plot_data[column].mean(), plot_data[column].std())
                        fig.add_trace(go.Scatter(
                            x=x_range, y=pdf * len(plot_data) * (plot_data[column].max() - plot_data[column].min()) / bins,
                            mode='lines', name='Normal Distribution', line=dict(color='red', width=2)
                        ))
                    except:
                        pass
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical insights
                if len(plot_data) > 0:
                    stats_data = plot_data[column].describe()
                    skewness = plot_data[column].skew()
                    st.write(f"**Statistics**: Mean = {stats_data['mean']:.2f}, Std = {stats_data['std']:.2f}, Skewness = {skewness:.2f}")
                    
                    if abs(skewness) > 1:
                        st.info(f"Distribution is {'right' if skewness > 0 else 'left'}-skewed")
                    else:
                        st.info("Distribution is approximately symmetric")
                
            except Exception as e:
                st.error(f"Error generating histogram: {str(e)}")
    else:
        st.warning("No numeric columns available for histogram analysis")

def box_plot_analysis(df):
    """Interactive box plot analysis"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols and categorical_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            numeric_col = st.selectbox("Numeric Column", numeric_cols, key="box_numeric")
        with col2:
            # Filter categorical columns with reasonable number of categories
            suitable_categorical = [col for col in categorical_cols if 1 < df[col].nunique() <= 20]
            if suitable_categorical:
                categorical_col = st.selectbox("Categorical Column", suitable_categorical, key="box_categorical")
            else:
                st.warning("No suitable categorical columns (need 2-20 unique values)")
                return
        
        if numeric_col and categorical_col:
            try:
                plot_data = df[[numeric_col, categorical_col]].dropna()
                
                fig = px.box(
                    plot_data, 
                    x=categorical_col, 
                    y=numeric_col,
                    color=categorical_col,
                    title=f"Distribution of {numeric_col} by {categorical_col}",
                    points="suspectedoutliers"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical insights
                groups = [group for name, group in plot_data.groupby(categorical_col)[numeric_col]]
                if len(groups) > 1:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        st.info(f"**ANOVA Test**: F-statistic = {f_stat:.3f}, p-value = {p_value:.3f}")
                        
                        if p_value < 0.05:
                            st.success("Significant differences found between groups (p < 0.05)")
                        else:
                            st.info("No significant differences between groups (p ≥ 0.05)")
                    except:
                        st.info("Could not perform ANOVA test")
                
                # Group statistics
                st.write("**Group Statistics:**")
                group_stats = plot_data.groupby(categorical_col)[numeric_col].agg(['mean', 'std', 'count']).round(2)
                st.dataframe(group_stats)
                    
            except Exception as e:
                st.error(f"Error generating box plot: {str(e)}")
    else:
        st.warning("Need both numeric and categorical columns for box plot analysis")

def correlation_analysis(df):
    """Enhanced correlation analysis"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        # Correlation matrix
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        st.write("**🔍 Strong Correlations (|r| > 0.7)**")
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        if strong_corrs:
            for col1, col2, corr in strong_corrs:
                st.write(f"- **{col1}** ↔ **{col2}**: {corr:.3f}")
        else:
            st.info("No strong correlations found (|r| > 0.7)")
        
        # Weak correlations
        st.write("**📉 Weak Correlations (|r| < 0.1)**")
        weak_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) < 0.1:
                    weak_corrs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        if weak_corrs:
            for col1, col2, corr in weak_corrs[:5]:  # Show top 5
                st.write(f"- **{col1}** ↔ **{col2}**: {corr:.3f}")
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

def time_series_analysis(df):
    """Time series analysis for datetime columns"""
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if datetime_cols and numeric_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            date_col = st.selectbox("Date Column", datetime_cols, key="ts_date")
        with col2:
            value_col = st.selectbox("Value Column", numeric_cols, key="ts_value")
        
        if date_col and value_col:
            try:
                # Resample options
                resample_freq = st.selectbox(
                    "Resampling Frequency",
                    ["Raw", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                    key="ts_freq"
                )
                
                freq_map = {
                    "Raw": None,
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "M",
                    "Quarterly": "Q",
                    "Yearly": "Y"
                }
                
                # Create time series
                ts_data = df[[date_col, value_col]].dropna()
                ts_data = ts_data.set_index(date_col).sort_index()
                
                if resample_freq != "Raw" and freq_map[resample_freq]:
                    ts_data = ts_data[value_col].resample(freq_map[resample_freq]).mean().reset_index()
                    ts_data.columns = [date_col, value_col]
                
                fig = px.line(
                    ts_data,
                    x=date_col,
                    y=value_col,
                    title=f"Time Series: {value_col} over Time",
                    labels={value_col: value_col, date_col: 'Date'}
                )
                
                # Add trend line
                if len(ts_data) > 1 and resample_freq != "Raw":
                    try:
                        x_numeric = np.arange(len(ts_data))
                        z = np.polyfit(x_numeric, ts_data[value_col], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=ts_data[date_col],
                            y=p(x_numeric),
                            mode='lines',
                            name='Trend Line',
                            line=dict(color='red', dash='dash')
                        ))
                    except:
                        pass
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series statistics
                if len(ts_data) > 0:
                    st.write("**Time Series Statistics:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Start Date", ts_data[date_col].min().strftime('%Y-%m-%d'))
                    with col2:
                        st.metric("End Date", ts_data[date_col].max().strftime('%Y-%m-%d'))
                    with col3:
                        st.metric("Data Points", len(ts_data))
                
            except Exception as e:
                st.error(f"Error generating time series: {str(e)}")
    else:
        st.warning("Need both datetime and numeric columns for time series analysis")

def categorical_analysis(df):
    """Categorical data analysis"""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        selected_col = st.selectbox("Select Categorical Column", categorical_cols, key="cat_analysis")
        
        if selected_col:
            # Value counts and proportions
            value_counts = df[selected_col].value_counts()
            proportions = df[selected_col].value_counts(normalize=True) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Value Counts (Top 10)**")
                for value, count in value_counts.head(10).items():
                    st.write(f"- {value}: {count:,} ({proportions[value]:.1f}%)")
            
            with col2:
                # Visualization
                top_values = value_counts.head(10)
                if not top_values.empty:
                    fig = px.pie(
                        values=top_values.values,
                        names=top_values.index,
                        title=f"Distribution of {selected_col} (Top 10)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Additional statistics
            st.write("**Column Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Unique Values", value_counts.shape[0])
            with col2:
                st.metric("Most Frequent", value_counts.index[0])
            with col3:
                st.metric("Mode Count", value_counts.iloc[0])
            with col4:
                mode_percentage = (value_counts.iloc[0] / len(df)) * 100
                st.metric("Mode %", f"{mode_percentage:.1f}%")
            
            # Check for high cardinality
            if value_counts.shape[0] > 50:
                st.warning(f"High cardinality: {value_counts.shape[0]} unique values. Consider grouping or encoding.")
    else:
        st.warning("No categorical columns available for analysis")

# ===============================
# ENHANCED EXPORT FUNCTIONS
# ===============================

def generate_enhanced_data_profile(df, include_charts=True, include_statistics=True):
    """Generate professional data profile with comprehensive analysis"""
    
    # Calculate comprehensive metrics
    total_rows = len(df)
    total_columns = len(df.columns)
    total_cells = total_rows * total_columns
    missing_total = df.isnull().sum().sum()
    completeness = (1 - missing_total / total_cells) * 100 if total_cells > 0 else 100
    duplicate_count = df.duplicated().sum()
    memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
    
    # Data composition
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
    datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
    bool_cols = len(df.select_dtypes(include=['bool']).columns)
    
    # Generate charts if requested
    charts_html = ""
    if include_charts:
        try:
            # Data types chart
            dtype_counts = df.dtypes.astype(str).value_counts()
            if not dtype_counts.empty:
                fig = px.pie(values=dtype_counts.values, names=dtype_counts.index, 
                           title="Data Types Distribution")
                charts_html += fig.to_html(include_plotlyjs='cdn', div_id="dtype_chart")
            
            # Missing values chart
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if not missing_data.empty:
                fig = px.bar(x=missing_data.index, y=missing_data.values, 
                           title="Missing Values by Column")
                charts_html += fig.to_html(include_plotlyjs=False, div_id="missing_chart")
        except Exception as e:
            charts_html += f"<!-- Chart generation error: {str(e)} -->"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Professional Data Profile Report</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                line-height: 1.6;
                color: #333;
                background: #f8f9fa;
                min-height: 100vh;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
                border-radius: 20px;
                overflow: hidden;
                margin-top: 30px;
                margin-bottom: 30px;
            }}
            
            .header {{
                background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
                color: white;
                padding: 50px 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.8rem;
                font-weight: 700;
                margin-bottom: 10px;
            }}
            
            .metadata {{
                background: #f8f9fa;
                padding: 30px 40px;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: white;
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
                text-align: center;
                border-left: 4px solid #3498db;
            }}
            
            .stat-value {{
                font-size: 2.2rem;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                font-size: 0.9rem;
                color: #6c757d;
                font-weight: 500;
                text-transform: uppercase;
            }}
            
            .section {{
                padding: 40px;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .section-title {{
                font-size: 1.5rem;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 25px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }}
            
            .data-table th,
            .data-table td {{
                padding: 15px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }}
            
            .data-table th {{
                background: #3498db;
                color: white;
                font-weight: 600;
            }}
            
            .quality-score {{
                display: inline-block;
                padding: 8px 16px;
                background: #00b894;
                color: white;
                border-radius: 20px;
                font-weight: 600;
            }}
            
            .insight-card {{
                background: #3498db;
                color: white;
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 20px;
            }}
            
            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }}
            
            .footer {{
                background: #2c3e50;
                color: white;
                text-align: center;
                padding: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Professional Data Profile Report</h1>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            </div>
            
            <div class="metadata">
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{total_rows:,}</div>
                        <div class="stat-label">Total Rows</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{total_columns}</div>
                        <div class="stat-label">Total Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{memory_usage:.2f} MB</div>
                        <div class="stat-label">Memory Usage</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{completeness:.1f}%</div>
                        <div class="stat-label">Data Quality</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📈 Dataset Overview</div>
                <p><strong>Total Cells:</strong> {total_cells:,}</p>
                <p><strong>Missing Values:</strong> {missing_total:,} ({missing_total/total_cells*100:.2f}%)</p>
                <p><strong>Duplicate Rows:</strong> {duplicate_count} ({duplicate_count/total_rows*100:.2f}%)</p>
            </div>
            
            <div class="section">
                <div class="section-title">🔍 Data Composition</div>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{numeric_cols}</div>
                        <div class="stat-label">Numeric Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{categorical_cols}</div>
                        <div class="stat-label">Categorical Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{datetime_cols}</div>
                        <div class="stat-label">Datetime Columns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{bool_cols}</div>
                        <div class="stat-label">Boolean Columns</div>
                    </div>
                </div>
            </div>
            
            {charts_html}
            
            <div class="section">
                <div class="section-title">📋 Column Details</div>
                {create_detailed_column_table(df)}
            </div>
            
            <div class="section">
                <div class="section-title">💡 Key Insights & Recommendations</div>
                <div class="insight-card">
                    <h3>🎯 Data Quality Assessment</h3>
                    <p>• Overall data quality: <span class="quality-score">{completeness:.1f}/100</span></p>
                    <p>• Memory optimization potential: {suggest_memory_optimization(df):.1f}%</p>
                    <p>• Data cleaning priority: {'HIGH' if missing_total > total_cells * 0.1 else 'MEDIUM' if missing_total > total_cells * 0.05 else 'LOW'}</p>
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by GenAI Data Analyst Pro | Enterprise Data Intelligence</p>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_professional_quality_report(df, report_format="Comprehensive HTML"):
    """Generate professional quality report with format options"""
    quality_score = calculate_data_quality_score(df)
    issues = identify_data_quality_issues(df)
    
    # Calculate additional metrics
    total_cells = len(df) * len(df.columns)
    missing_total = df.isnull().sum().sum()
    completeness = (1 - missing_total / total_cells) * 100 if total_cells > 0 else 100
    duplicate_count = df.duplicated().sum()
    
    if report_format == "Executive Summary":
        issues_display = "".join([f'<div class="issue-item">{issue}</div>' for issue in issues[:5]]) if issues else '<p>No major quality issues detected</p>'
    else:
        issues_display = "".join([f'<div class="issue-item">{issue}</div>' for issue in issues]) if issues else '<p>No major quality issues detected</p>'
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data Quality Assessment Report - {report_format}</title>
        <style>
            body {{
                font-family: 'Inter', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: linear-gradient(135deg, #2c3e50, #3498db);
                color: white;
                border-radius: 15px;
            }}
            .quality-score {{
                font-size: 4rem;
                font-weight: bold;
                color: {'#00b894' if quality_score >= 80 else '#f39c12' if quality_score >= 60 else '#e74c3c'};
                margin: 20px 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border-left: 4px solid #3498db;
            }}
            .metric-value {{
                font-size: 1.8rem;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-label {{
                font-size: 0.9rem;
                color: #6c757d;
                margin-top: 5px;
            }}
            .section {{
                margin: 40px 0;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 15px;
            }}
            .issue-item {{
                padding: 20px;
                margin: 15px 0;
                background: white;
                border-radius: 10px;
                border-left: 4px solid #e74c3c;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            }}
            .recommendation {{
                background: #d4edda;
                border-left: 4px solid #28a745;
                padding: 20px;
                margin: 15px 0;
                border-radius: 10px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background: #2c3e50;
                color: white;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Data Quality Assessment Report</h1>
                <p>{report_format} - Generated on {datetime.now().strftime('%B %d, %Y')}</p>
                <div class="quality-score">{quality_score}/100</div>
                <p>Overall Data Quality Score</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{completeness:.1f}%</div>
                    <div class="metric-label">Completeness</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{duplicate_count}</div>
                    <div class="metric-label">Duplicates</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(df.columns)}</div>
                    <div class="metric-label">Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(df):,}</div>
                    <div class="metric-label">Rows</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Quality Issues Found</h2>
                {issues_display}
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendation">
                    <h3>Immediate Actions</h3>
                    <p>• Address high-missing value columns first</p>
                    <p>• Remove or investigate duplicate records</p>
                    <p>• Consider data type optimization for memory efficiency</p>
                </div>
                {'<div class="recommendation"><h3>Additional Considerations</h3><p>• Review high-cardinality categorical variables</p><p>• Validate datetime conversions</p></div>' if report_format != "Executive Summary" else ''}
            </div>
            
            <div class="footer">
                <p>Generated by Advanced Data Analytics Suite | Quality Assessment Module</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_analysis_dashboard(df, include_charts=True, include_samples=True):
    """Generate comprehensive analysis dashboard"""
    
    # Calculate metrics
    total_rows = len(df)
    total_columns = len(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Sample data for inclusion
    sample_html = ""
    if include_samples and len(df) > 0:
        sample_data = df.head(5).to_html(classes='data-table', index=False, escape=False)
        sample_html = f"""
        <div class="section">
            <div class="section-title">📋 Data Sample</div>
            {sample_data}
        </div>
        """
    
    # Charts section
    charts_html = ""
    if include_charts:
        try:
            # Numeric columns summary
            if len(numeric_cols) > 0:
                stats = df[numeric_cols].describe()
                charts_html += f"""
                <div class="section">
                    <div class="section-title">📊 Numeric Columns Summary</div>
                    {stats.to_html(classes='data-table', float_format='%.2f')}
                </div>
                """
        except Exception as e:
            charts_html += f"<!-- Error generating charts: {str(e)} -->"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Advanced Analysis Dashboard</title>
        <style>
            body {{
                font-family: 'Inter', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .dashboard {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 30px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border-radius: 15px;
            }}
            .metrics {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }}
            .metric {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                text-align: center;
                border-left: 4px solid #3498db;
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-label {{
                font-size: 0.9rem;
                color: #6c757d;
                margin-top: 10px;
            }}
            .section {{
                margin: 40px 0;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 15px;
            }}
            .section-title {{
                font-size: 1.4rem;
                font-weight: 600;
                color: #2c3e50;
                margin-bottom: 20px;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            }}
            .data-table th,
            .data-table td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e9ecef;
            }}
            .data-table th {{
                background: #3498db;
                color: white;
                font-weight: 600;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background: #2c3e50;
                color: white;
                border-radius: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="header">
                <h1>Advanced Analysis Dashboard</h1>
                <p>Comprehensive Data Analysis Report</p>
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <div class="metric-value">{total_rows:,}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{total_columns}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(numeric_cols)}</div>
                    <div class="metric-label">Numeric Columns</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(categorical_cols)}</div>
                    <div class="metric-label">Categorical Columns</div>
                </div>
            </div>
            
            {sample_html}
            {charts_html}
            
            <div class="section">
                <div class="section-title">📈 Key Insights</div>
                <div style="background: white; padding: 20px; border-radius: 10px;">
                    {generate_insights_text(df)}
                </div>
            </div>
            
            <div class="footer">
                <p>Generated by GenAI Data Analyst Pro | Analysis Dashboard v2.0</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def export_enhanced_csv(df):
    """Export DataFrame to CSV with enhanced formatting"""
    output = io.StringIO()
    
    # Add metadata header
    output.write(f"# Data Export Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output.write(f"# Total Rows: {len(df):,}, Total Columns: {len(df.columns)}\n")
    output.write(f"# Missing Values: {df.isnull().sum().sum():,}\n")
    output.write("# \n")
    
    # Export data
    df.to_csv(output, index=False)
    
    return output.getvalue()

def generate_complete_analysis_suite(df, export_name):
    """Generate a complete analysis suite as ZIP file"""
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 1. Data profile
        profile_html = generate_enhanced_data_profile(df, True, True)
        zip_file.writestr(f"{export_name}_profile.html", profile_html)
        
        # 2. Quality report
        quality_html = generate_professional_quality_report(df, "Comprehensive HTML")
        zip_file.writestr(f"{export_name}_quality_report.html", quality_html)
        
        # 3. Analysis dashboard
        dashboard_html = generate_analysis_dashboard(df, True, True)
        zip_file.writestr(f"{export_name}_dashboard.html", dashboard_html)
        
        # 4. Raw data (CSV)
        csv_data = export_enhanced_csv(df)
        zip_file.writestr(f"{export_name}_data.csv", csv_data)
        
        # 5. Metadata JSON
        metadata = {
            "export_timestamp": datetime.now().isoformat(),
            "dataset_name": export_name,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024**2),
            "completeness_score": calculate_data_quality_score(df),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())
        }
        zip_file.writestr(f"{export_name}_metadata.json", json.dumps(metadata, indent=2))
        
        # 6. Readme file
        readme_content = f"""
        COMPLETE ANALYSIS SUITE - {export_name}
        ===================================
        
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Contents:
        1. {export_name}_profile.html - Comprehensive data profile
        2. {export_name}_quality_report.html - Detailed quality assessment
        3. {export_name}_dashboard.html - Interactive analysis dashboard
        4. {export_name}_data.csv - Cleaned dataset in CSV format
        5. {export_name}_metadata.json - Dataset metadata and metrics
        
        Dataset Summary:
        - Rows: {len(df):,}
        - Columns: {len(df.columns)}
        - Memory Usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB
        - Data Quality Score: {calculate_data_quality_score(df):.1f}/100
        
        For questions or support, contact your data analytics team.
        """
        zip_file.writestr("README.txt", readme_content)
    
    buffer.seek(0)
    return buffer

def generate_executive_summary(df):
    """Generate an executive summary report"""
    
    quality_score = calculate_data_quality_score(df)
    total_rows = len(df)
    total_columns = len(df.columns)
    missing_total = df.isnull().sum().sum()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Executive Summary - Data Analysis</title>
        <style>
            body {{
                font-family: 'Inter', Arial, sans-serif;
                margin: 0;
                padding: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .executive-container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 50px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.2);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 30px;
                border-bottom: 2px solid #e9ecef;
            }}
            .score {{
                font-size: 3rem;
                font-weight: bold;
                color: {'#00b894' if quality_score >= 80 else '#f39c12' if quality_score >= 60 else '#e74c3c'};
                margin: 20px 0;
            }}
            .highlight {{
                background: #fff3cd;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 4px solid #ffc107;
            }}
            .metric {{
                display: flex;
                justify-content: space-between;
                margin: 15px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #e9ecef;
                color: #6c757d;
            }}
        </style>
    </head>
    <body>
        <div class="executive-container">
            <div class="header">
                <h1>Executive Summary</h1>
                <p>Data Quality Assessment</p>
                <div class="score">{quality_score}/100</div>
                <p>Overall Data Quality Score</p>
            </div>
            
            <div class="highlight">
                <h3>📊 Key Findings</h3>
                <div class="metric">
                    <span>Dataset Size:</span>
                    <strong>{total_rows:,} rows × {total_columns} columns</strong>
                </div>
                <div class="metric">
                    <span>Data Completeness:</span>
                    <strong>{(1 - missing_total/(total_rows*total_columns))*100:.1f}%</strong>
                </div>
                <div class="metric">
                    <span>Critical Issues:</span>
                    <strong>{len(identify_data_quality_issues(df))}</strong>
                </div>
            </div>
            
            <h3>🎯 Recommendations</h3>
            <ul>
                <li>Review data quality issues for immediate action</li>
                <li>Consider data type optimization for performance</li>
                <li>Validate business rules against data patterns</li>
                <li>Schedule regular data quality assessments</li>
            </ul>
            
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
                <p>Confidential - For Executive Review Only</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_insights_text(df):
    """Generate textual insights for reports"""
    insights = get_enhanced_data_overview_insights(df)
    insights_html = ""
    
    for insight in insights[:6]:  # Top 6 insights
        icon = "✅" if "✅" in insight or "Success" in insight else "⚠️" if "Warning" in insight or "🚨" in insight else "ℹ️"
        clean_insight = insight.replace("✅", "").replace("🚨", "").replace("⚠️", "").strip()
        insights_html += f'<p>{icon} {clean_insight}</p>'
    
    return insights_html

# ===============================
# UTILITY FUNCTIONS
# ===============================

def auto_detect_and_convert_datetime(df):
    """Automatically detect and convert datetime columns"""
    datetime_columns = []
    
    for col in df.columns:
        # Skip if already datetime
        if 'datetime' in str(df[col].dtype):
            continue
            
        # Check column name patterns for datetime indicators
        datetime_keywords = ['time', 'date', 'timestamp', 'dt', 'datetime', 'created', 'updated', 'modified']
        
        if any(keyword in col.lower() for keyword in datetime_keywords):
            original_dtype = str(df[col].dtype)
            
            # Try to convert to datetime
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                # Check if conversion was successful (not all NaT)
                if not converted.isna().all():
                    # Check if we have a reasonable number of successful conversions
                    success_rate = (1 - converted.isna().sum() / len(converted)) * 100
                    if success_rate > 50:  # More than 50% successful conversions
                        df[col] = converted
                        datetime_columns.append((col, original_dtype, 'datetime64[ns]'))
            except Exception:
                continue
    
    return df, datetime_columns

def display_dataframe_safe(df, description=""):
    """Safely display dataframe handling Arrow serialization issues"""
    try:
        if len(df) == 0:
            st.info("No data to display")
            return
            
        # Try to display normally first
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.warning(f"Display optimization issue with {description}. Showing raw data...")
        # Fallback: convert problematic columns to string
        df_fixed = df.copy()
        for col in df_fixed.columns:
            if df_fixed[col].dtype == 'object':
                # Convert object columns to string to avoid Arrow issues
                df_fixed[col] = df_fixed[col].astype(str)
        st.dataframe(df_fixed, use_container_width=True)

def create_enhanced_column_info_table(df):
    """Create enhanced column information table"""
    if len(df.columns) == 0:
        return pd.DataFrame()
        
    column_info_data = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null_count = int(df[col].count())
        null_count = int(df[col].isnull().sum())
        unique_count = int(df[col].nunique())
        memory_usage = df[col].memory_usage(deep=True) / 1024  # KB
        
        # Data quality indicators
        completeness = (non_null_count / len(df)) * 100 if len(df) > 0 else 0
        uniqueness = (unique_count / len(df)) * 100 if len(df) > 0 else 0
        
        column_info_data.append({
            'Column': str(col),
            'Data Type': dtype,
            'Non-Null': non_null_count,
            'Null': null_count,
            'Unique': unique_count,
            'Completeness %': f"{completeness:.1f}%",
            'Memory (KB)': f"{memory_usage:.1f}"
        })
    
    return pd.DataFrame(column_info_data)

def create_dtype_counts(df):
    """Create data type counts with safe formatting"""
    if len(df.columns) == 0:
        return pd.DataFrame()
        
    dtype_counts = df.dtypes.astype(str).value_counts()
    if dtype_counts.empty:
        return pd.DataFrame()
    
    return pd.DataFrame({
        'count': dtype_counts.values
    }, index=dtype_counts.index.astype(str))

def create_enhanced_missing_data_summary(df):
    """Create enhanced missing data summary"""
    if len(df.columns) == 0:
        return pd.DataFrame()
        
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if missing_data.empty:
        return pd.DataFrame()
    
    missing_percentage = (missing_data / len(df)) * 100
    return pd.DataFrame({
        'missing_count': missing_data.values,
        'missing_percentage': missing_percentage.values
    }, index=missing_data.index.astype(str))

def get_enhanced_data_overview_insights(df):
    """Generate comprehensive automated insights"""
    insights = []
    
    if len(df) == 0:
        insights.append("🚨 Dataset is empty")
        return insights
    
    # Basic insights
    total_cells = len(df) * len(df.columns)
    insights.append(f"📊 Dataset contains {len(df):,} rows and {len(df.columns)} columns ({total_cells:,} total cells)")
    
    # Data quality insights
    completeness = (1 - df.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 100
    insights.append(f"✅ Data Completeness: {completeness:.1f}%")
    
    # Missing values insight
    missing_total = df.isnull().sum().sum()
    if missing_total > 0:
        missing_percentage = (missing_total / total_cells) * 100
        missing_cols = len(df.columns[df.isnull().any()])
        insights.append(f"⚠️ {missing_total:,} missing values ({missing_percentage:.1f}%) across {missing_cols} columns")
    
    # Duplicate insights
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        duplicate_percentage = (duplicate_count / len(df)) * 100
        insights.append(f"🔍 {duplicate_count:,} duplicate rows ({duplicate_percentage:.1f}% of data)")
    
    # Data types insight
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    datetime_cols = df.select_dtypes(include=['datetime64']).columns
    bool_cols = df.select_dtypes(include=['bool']).columns
    
    type_breakdown = []
    if len(numeric_cols) > 0:
        type_breakdown.append(f"{len(numeric_cols)} numeric")
    if len(categorical_cols) > 0:
        type_breakdown.append(f"{len(categorical_cols)} categorical")
    if len(datetime_cols) > 0:
        type_breakdown.append(f"{len(datetime_cols)} datetime")
    if len(bool_cols) > 0:
        type_breakdown.append(f"{len(bool_cols)} boolean")
    
    if type_breakdown:
        insights.append(f"🎯 Data Composition: {', '.join(type_breakdown)}")
    
    # High cardinality warning (only for non-datetime categorical)
    non_datetime_categorical = [col for col in categorical_cols 
                               if not any(keyword in col.lower() for keyword in 
                                        ['time', 'date', 'timestamp', 'dt', 'datetime'])]
    high_cardinality = [col for col in non_datetime_categorical if df[col].nunique() > 50]
    if high_cardinality:
        insights.append(f"🚨 {len(high_cardinality)} high-cardinality columns (>50 unique values) may need encoding")
    
    # Constant columns warning
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        insights.append(f"🚨 {len(constant_cols)} constant columns (single value) provide no predictive power")
    
    # Memory insights
    current_memory = df.memory_usage(deep=True).sum() / (1024**2)
    optimized_memory = estimate_optimized_memory(df)
    savings = current_memory - optimized_memory
    if savings > 0.1:  # Only show if savings > 0.1 MB
        insights.append(f"💾 Memory optimization potential: {savings:.1f} MB ({savings/current_memory*100:.1f}% reduction)")
    
    # Datetime insights
    if len(datetime_cols) > 0:
        insights.append(f"🕒 {len(datetime_cols)} datetime columns detected - Ready for time series analysis")
    
    return insights

def suggest_memory_optimization(df):
    """Calculate memory optimization potential percentage"""
    if len(df) == 0:
        return 0
        
    current_memory = df.memory_usage(deep=True).sum()
    optimized_memory = estimate_optimized_memory(df)
    
    if current_memory > 0:
        return ((current_memory - optimized_memory) / current_memory) * 100
    return 0

def estimate_optimized_memory(df):
    """Estimate memory usage after optimization"""
    if len(df) == 0:
        return 0
        
    optimized_memory = 0
    
    for col in df.columns:
        col_data = df[col]
        current_dtype = col_data.dtype
        
        # Simple estimation - in real implementation, you'd calculate actual optimized memory
        if current_dtype == 'object':
            # String columns could be converted to category if low cardinality
            if col_data.nunique() / len(col_data) < 0.5:  # Low cardinality
                optimized_memory += len(col_data) * 4  # Approximate category size
            else:
                optimized_memory += col_data.memory_usage(deep=True)
        elif current_dtype == 'int64':
            optimized_memory += len(col_data) * 4  # Could use int32
        elif current_dtype == 'float64':
            optimized_memory += len(col_data) * 4  # Could use float32
        else:
            optimized_memory += col_data.memory_usage(deep=True)
    
    return optimized_memory

def suggest_data_type_optimization(df):
    """Suggest optimal data types for memory efficiency"""
    suggestions = []
    
    for col in df.columns:
        col_data = df[col]
        current_dtype = str(col_data.dtype)
        current_memory = col_data.memory_usage(deep=True) / 1024**2  # MB
        
        suggested_type = None
        reason = ""
        
        if current_dtype == 'object':
            # Skip datetime-like columns that were converted
            datetime_keywords = ['time', 'date', 'timestamp', 'dt', 'datetime']
            if not any(keyword in col.lower() for keyword in datetime_keywords):
                unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
                if unique_ratio < 0.5:  # Low cardinality
                    suggested_type = 'category'
                    reason = f"Low cardinality ({col_data.nunique()} unique values)"
        
        elif current_dtype == 'int64':
            if not col_data.isnull().all() and len(col_data) > 0:
                min_val = col_data.min()
                max_val = col_data.max()
                
                if min_val >= -128 and max_val <= 127:
                    suggested_type = 'int8'
                elif min_val >= -32768 and max_val <= 32767:
                    suggested_type = 'int16'
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    suggested_type = 'int32'
                else:
                    suggested_type = 'int64'
                
                reason = f"Value range: {min_val} to {max_val}"
        
        elif current_dtype == 'float64':
            suggested_type = 'float32'
            reason = "Precision reduction acceptable for most use cases"
        
        if suggested_type and suggested_type != current_dtype:
            # Estimate memory saving (simplified)
            estimated_saving = current_memory * 0.5  # Simplified estimation
            
            suggestions.append({
                'column': col,
                'current_type': current_dtype,
                'suggested_type': suggested_type,
                'memory_saving_mb': estimated_saving,
                'reason': reason
            })
    
    # Sort by memory saving (descending)
    suggestions.sort(key=lambda x: x['memory_saving_mb'], reverse=True)
    return suggestions

def identify_data_quality_issues(df):
    """Identify and report data quality issues with proper datetime handling"""
    issues = []
    
    if len(df) == 0:
        issues.append("Dataset is empty")
        return issues
    
    # Missing values issues
    high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.5]
    for col in high_missing_cols:
        missing_pct = (df[col].isnull().sum() / len(df)) * 100
        issues.append(f"**{col}**: {missing_pct:.1f}% missing values - Consider removal or imputation")
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    for col in constant_cols:
        if len(df) > 0:
            issues.append(f"**{col}**: Constant value '{df[col].iloc[0]}' - No predictive value")
    
    # High cardinality categorical (EXCLUDE datetime columns)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Exclude columns that are actually datetime but stored as object
    datetime_pattern_columns = []
    for col in categorical_cols:
        # Check if column name suggests it's datetime/timestamp
        datetime_keywords = ['time', 'date', 'timestamp', 'dt', 'datetime', 'created', 'updated']
        if any(keyword in col.lower() for keyword in datetime_keywords):
            datetime_pattern_columns.append(col)
            continue  # Skip datetime-like columns
    
    # Now check for high cardinality only in non-datetime categorical columns
    non_datetime_categorical = [col for col in categorical_cols if col not in datetime_pattern_columns]
    high_cardinality = [col for col in non_datetime_categorical if df[col].nunique() > 100]
    
    for col in high_cardinality:
        issues.append(f"**{col}**: {df[col].nunique()} unique values - May need encoding or grouping")
    
    # Suggest datetime conversion for timestamp-like columns that are still objects
    for col in datetime_pattern_columns:
        if df[col].nunique() > 50:  # High unique values suggest it's actually datetime
            try:
                # Try to convert to datetime to confirm
                pd.to_datetime(df[col], errors='coerce')
                # If successful conversion with reasonable success rate
                temp_converted = pd.to_datetime(df[col], errors='coerce')
                success_rate = (1 - temp_converted.isna().sum() / len(temp_converted)) * 100
                if success_rate > 50:
                    issues.append(f"**{col}**: {df[col].nunique()} unique values - Consider converting to datetime format")
            except:
                # If conversion fails, it might actually be categorical
                if df[col].nunique() > 100:
                    issues.append(f"**{col}**: {df[col].nunique()} unique values - May need encoding or grouping")
    
    # Duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        duplicate_pct = (duplicate_count / len(df)) * 100
        issues.append(f"Duplicate Rows: {duplicate_count} ({duplicate_pct:.1f}%) - Consider removal")
    
    return issues

def calculate_data_quality_score(df):
    """Calculate overall data quality score"""
    if len(df) == 0:
        return 0
        
    total_cells = len(df) * len(df.columns)
    if total_cells == 0:
        return 0
        
    completeness = (1 - df.isnull().sum().sum() / total_cells) * 100
    duplicate_score = max(0, 100 - (df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 100
    constant_cols_penalty = len([col for col in df.columns if df[col].nunique() == 1]) * 5
    
    return max(0, min(100, (completeness + duplicate_score) / 2 - constant_cols_penalty))

def create_detailed_column_table(df):
    """Create detailed column analysis table for reports"""
    if len(df.columns) == 0:
        return "<p>No columns available</p>"
        
    table_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_count = df[col].isnull().sum()
        unique_count = df[col].nunique()
        completeness = (non_null / len(df)) * 100 if len(df) > 0 else 0
        
        table_rows.append(f"""
        <tr>
            <td><strong>{col}</strong></td>
            <td>{dtype}</td>
            <td>{non_null}</td>
            <td>{null_count}</td>
            <td>{unique_count}</td>
            <td>{completeness:.1f}%</td>
        </tr>
        """)
    
    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>Column</th>
                <th>Data Type</th>
                <th>Non-Null</th>
                <th>Null Count</th>
                <th>Unique Values</th>
                <th>Completeness</th>
            </tr>
        </thead>
        <tbody>
            {''.join(table_rows)}
        </tbody>
    </table>
    """