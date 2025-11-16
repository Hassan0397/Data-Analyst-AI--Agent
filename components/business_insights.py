import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import with error handling
try:
    from utils.business_kpi import BusinessKPIAnalyzer
except ImportError:
    st.error("BusinessKPIAnalyzer not found. Please ensure the business_kpi.py file is in the utils directory.")
    BusinessKPIAnalyzer = None

def render_business_insights():
    """Render enhanced business insights generation interface with user guidance"""
    
    st.header("💼 Advanced Business Insights & KPIs")
    
    # Check if data is available
    if 'processed_data' not in st.session_state or not st.session_state.processed_data:
        st.warning("📊 No data available. Please upload and process files first in the Data Upload section.")
        st.info("💡 Once you've uploaded your data, you'll be able to access comprehensive business insights here.")
        return
    
    if BusinessKPIAnalyzer is None:
        st.error("❌ Business analytics engine unavailable. Please check the backend setup.")
        return
    
    # User Guide Expansion Panel
    with st.expander("📚 **User Guide - Getting Started with Business Insights**", expanded=False):
        st.markdown("""
        ### Welcome to Business Insights!
        
        **For Non-Technical Users:** This tool helps you understand your business data through easy-to-use analytics.
        
        **Quick Start Guide:**
        1. **KPI Dashboard**: View key business metrics and performance indicators
        2. **Trend Analysis**: Analyze how metrics change over time
        3. **Performance Metrics**: Compare segments and identify top performers
        4. **Comparative Analysis**: Benchmark against standards and find gaps
        5. **Strategic Insights**: Get AI-powered recommendations
        
        **Tips:**
        - Start with KPI Dashboard to get overview
        - Use Trend Analysis to spot patterns
        - Check Strategic Insights for actionable advice
        - Export any chart by clicking the camera icon 📷
        """)
    
    selected_file = st.session_state.get('current_file', '')
    if not selected_file or selected_file not in st.session_state.processed_data:
        st.error("No valid file selected. Please select a file from the upload section.")
        return
    
    data_info = st.session_state.processed_data[selected_file]
    df = data_info['dataframe']
    
    # Data validation
    if df.empty:
        st.error("Selected file contains no data. Please upload a valid dataset.")
        return
    
    try:
        kpi_analyzer = BusinessKPIAnalyzer(df)
    except Exception as e:
        st.error(f"Error initializing analytics engine: {str(e)}")
        return
    
    # Enhanced business insights tabs
    business_tabs = st.tabs([
        "📊 KPI Dashboard",
        "📈 Trend Analysis", 
        "🎯 Performance Metrics",
        "📊 Comparative Analysis",
        "🔍 Driver Analysis",
        "💡 Strategic Insights",
        "📋 Executive Summary"
    ])
    
    with business_tabs[0]:
        render_enhanced_kpi_dashboard(kpi_analyzer)
    
    with business_tabs[1]:
        render_enhanced_trend_analysis(kpi_analyzer)
    
    with business_tabs[2]:
        render_enhanced_performance_metrics(kpi_analyzer)
    
    with business_tabs[3]:
        render_enhanced_comparative_analysis(kpi_analyzer)
    
    with business_tabs[4]:
        render_driver_analysis(kpi_analyzer)
    
    with business_tabs[5]:
        render_enhanced_strategic_insights(kpi_analyzer)
    
    with business_tabs[6]:
        render_executive_summary(kpi_analyzer)

def render_enhanced_kpi_dashboard(kpi_analyzer):
    """Render enhanced KPI dashboard with comprehensive metrics"""
    
    with st.expander("💡 **KPI Dashboard Guide**", expanded=False):
        st.markdown("""
        **Understanding KPI Dashboard:**
        - **Primary KPIs**: Your main business metrics (revenue, sales, etc.)
        - **Growth Metrics**: How metrics are changing over time
        - **Distribution Metrics**: How values are spread across your data
        - **Quality Metrics**: Data reliability and completeness
        
        **How to use:**
        1. Select your main value column (like revenue or sales)
        2. Choose quantity if you have volume data
        3. Pick date column for growth calculations
        4. View automatic KPI calculations
        """)
    
    st.subheader("📊 Advanced KPI Dashboard")
    
    df = kpi_analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.info("ℹ️ No numeric columns found for KPI analysis. Please ensure your data contains numerical values.")
        return
    
    # Enhanced KPI configuration
    st.write("#### 🛠️ Configure Your Key Performance Indicators")
    
    kpi_config_col1, kpi_config_col2 = st.columns(2)
    
    with kpi_config_col1:
        revenue_col = st.selectbox(
            "💰 Revenue/Value Column (Primary KPI)", 
            [None] + numeric_cols,
            help="Select your main business metric (sales, revenue, etc.)",
            key="kpi_revenue"
        )
        quantity_col = st.selectbox(
            "📦 Quantity/Volume Column", 
            [None] + numeric_cols,
            help="Select quantity column if available",
            key="kpi_quantity"
        )
        cost_col = st.selectbox(
            "💸 Cost/Expense Column", 
            [None] + numeric_cols,
            help="Select cost column for profitability analysis",
            key="kpi_cost"
        )
    
    with kpi_config_col2:
        date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        date_col = st.selectbox(
            "📅 Date Column (for trends)", 
            [None] + date_cols,
            help="Select date column for growth calculations",
            key="kpi_date"
        )
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        category_col = st.selectbox(
            "🏷️ Category/Segment Column", 
            [None] + categorical_cols,
            help="Select category for segmentation analysis",
            key="kpi_category"
        )
        
        customer_col = st.selectbox(
            "👥 Customer ID Column", 
            [None] + df.columns.tolist(),
            help="Select customer identifier for customer analytics",
            key="kpi_customer"
        )
    
    if not revenue_col:
        st.info("👆 Please select a revenue/value column to start KPI analysis")
        return
    
    try:
        # Calculate comprehensive KPIs
        kpis = kpi_analyzer.calculate_comprehensive_kpis(
            revenue_col, quantity_col, cost_col, date_col, customer_col
        )
        
        if not kpis:
            st.error("Failed to calculate KPIs. Please check your data and column selections.")
            return
        
        # Display Enhanced KPI cards in multiple rows
        st.write("#### 📈 Key Performance Indicators")
        
        # Row 1: Core Business Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_rev = kpis.get('total_revenue', 0)
            revenue_growth = kpis.get('revenue_growth')
            st.metric(
                "Total Revenue", 
                f"${total_rev:,.2f}" if total_rev >= 0 else f"-${abs(total_rev):,.2f}",
                delta=f"{revenue_growth:+.1f}%" if revenue_growth is not None else None,
                help="Total value of selected revenue column",
                delta_color="normal" if (revenue_growth or 0) >= 0 else "inverse"
            )
        
        with col2:
            avg_val = kpis.get('average_value', 0)
            avg_growth = kpis.get('avg_value_growth')
            st.metric(
                "Average Value", 
                f"${avg_val:,.2f}",
                delta=f"{avg_growth:+.1f}%" if avg_growth is not None else None,
                help="Average value per transaction/record",
                delta_color="normal" if (avg_growth or 0) >= 0 else "inverse"
            )
        
        with col3:
            total_qty = kpis.get('total_quantity')
            qty_growth = kpis.get('quantity_growth')
            if total_qty is not None:
                st.metric(
                    "Total Quantity", 
                    f"{total_qty:,}",
                    delta=f"{qty_growth:+.1f}%" if qty_growth is not None else None,
                    help="Total quantity/volume",
                    delta_color="normal" if (qty_growth or 0) >= 0 else "inverse"
                )
            else:
                st.metric("Total Quantity", "N/A", help="Quantity column not selected")
        
        with col4:
            txn_count = kpis.get('transaction_count', 0)
            txn_growth = kpis.get('transaction_growth')
            st.metric(
                "Transactions", 
                f"{txn_count:,}",
                delta=f"{txn_growth:+.1f}%" if txn_growth is not None else None,
                help="Total number of records/transactions",
                delta_color="normal" if (txn_growth or 0) >= 0 else "inverse"
            )
        
        # Row 2: Financial Metrics
        if cost_col and kpis.get('total_profit') is not None:
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                profit = kpis.get('total_profit', 0)
                st.metric(
                    "Total Profit", 
                    f"${profit:,.2f}" if profit >= 0 else f"-${abs(profit):,.2f}",
                    help="Revenue minus costs",
                    delta_color="normal" if profit >= 0 else "inverse"
                )
            
            with col6:
                margin = kpis.get('profit_margin', 0)
                st.metric(
                    "Profit Margin", 
                    f"{margin:+.1f}%",
                    help="Profit as percentage of revenue",
                    delta_color="normal" if margin >= 0 else "inverse"
                )
            
            with col7:
                roi = kpis.get('roi', 0)
                st.metric(
                    "ROI", 
                    f"{roi:+.1f}%",
                    help="Return on investment",
                    delta_color="normal" if roi >= 0 else "inverse"
                )
            
            with col8:
                breakeven = kpis.get('breakeven_point', 0)
                if breakeven > 0:
                    st.metric(
                        "Breakeven Units", 
                        f"{breakeven:,.0f}",
                        help="Units needed to cover costs"
                    )
                else:
                    st.metric("Breakeven", "N/A", help="Insufficient data")
        
        # Row 3: Advanced Metrics
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            cv = kpis.get('coefficient_of_variation', 0)
            consistency_score = max(0, 100 - min(cv, 100))
            st.metric(
                "Consistency Score", 
                f"{consistency_score:.1f}%",
                help="Higher score = more consistent performance",
                delta_color="normal" if consistency_score >= 80 else "inverse"
            )
        
        with col10:
            gini = kpis.get('gini_coefficient', 0)
            equality_index = max(0, (1 - gini) * 100)
            st.metric(
                "Equality Index", 
                f"{equality_index:.1f}%",
                help="Higher = more equal distribution",
                delta_color="normal" if equality_index >= 50 else "inverse"
            )
        
        with col11:
            concentration = kpis.get('top_10_concentration', 0)
            st.metric(
                "Top 10% Concentration", 
                f"{concentration:.1f}%",
                help="% of value from top 10% items",
                delta_color="inverse" if concentration > 80 else "normal"
            )
        
        with col12:
            data_quality = kpis.get('data_quality_score', 100)
            st.metric(
                "Data Quality", 
                f"{data_quality:.1f}%",
                delta_color="inverse" if data_quality < 90 else "normal",
                help="Data completeness and reliability"
            )
        
        # Customer Metrics if available
        if customer_col and kpis.get('unique_customers'):
            st.write("#### 👥 Customer Metrics")
            col13, col14, col15, col16 = st.columns(4)
            
            with col13:
                customers = kpis.get('unique_customers', 0)
                st.metric("Unique Customers", f"{customers:,}")
            
            with col14:
                avg_customer_value = kpis.get('avg_customer_value', 0)
                st.metric("Avg Customer Value", f"${avg_customer_value:,.2f}")
            
            with col15:
                frequency = kpis.get('purchase_frequency', 0)
                st.metric("Purchase Frequency", f"{frequency:.2f}")
            
            with col16:
                clv = kpis.get('estimated_clv', 0)
                st.metric("Est. CLV", f"${clv:,.2f}")
        
        # Additional Visualizations
        st.write("#### 📊 Value Distribution")
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            try:
                fig_hist = px.histogram(df, x=revenue_col, title=f"Distribution of {revenue_col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.error(f"Could not create distribution chart: {str(e)}")
        
        with col_viz2:
            try:
                # Top categories if available
                if category_col and category_col in df.columns:
                    top_categories = df.groupby(category_col)[revenue_col].sum().nlargest(10)
                    fig_bar = px.bar(
                        x=top_categories.index, 
                        y=top_categories.values,
                        title=f"Top 10 {category_col} by {revenue_col}",
                        labels={'x': category_col, 'y': revenue_col}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            except Exception as e:
                st.info("Category visualization not available")
                
    except Exception as e:
        st.error(f"Error calculating KPIs: {str(e)}")
        st.info("Please check your column selections and data types.")

def render_enhanced_trend_analysis(kpi_analyzer):
    """Render enhanced trend analysis with forecasting"""
    
    with st.expander("💡 **Trend Analysis Guide**", expanded=False):
        st.markdown("""
        **Understanding Trend Analysis:**
        - **Trend Lines**: Shows how metrics change over time
        - **Growth Rates**: Calculates monthly/quarterly/yearly changes
        - **Seasonality**: Identifies recurring patterns
        - **Forecasting**: Predicts future trends
        
        **How to use:**
        1. Select date and value columns
        2. Choose time period (daily, weekly, monthly)
        3. Analyze growth patterns
        4. Check seasonality for recurring patterns
        """)
    
    st.subheader("📈 Advanced Trend Analysis")
    
    df = kpi_analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    
    if not date_cols:
        st.info("ℹ️ No date columns found for trend analysis. Please ensure your data contains date/datetime columns.")
        return
    
    if not numeric_cols:
        st.info("ℹ️ No numeric columns found for trend analysis.")
        return
    
    # Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        date_col = st.selectbox(
            "Select date column for trend analysis", 
            date_cols,
            key="trend_date"
        )
    
    with col2:
        value_col = st.selectbox(
            "Select value column for trend analysis", 
            numeric_cols,
            key="trend_value"
        )
    
    if not date_col or not value_col:
        st.info("👆 Please select both date and value columns to proceed")
        return
    
    try:
        # Enhanced time-based aggregation
        col3, col4 = st.columns(2)
        
        with col3:
            aggregation = st.selectbox(
                "Aggregation period",
                ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"],
                key="trend_agg"
            )
        
        with col4:
            show_forecast = st.checkbox("Show 3-month forecast", value=True)
            show_seasonality = st.checkbox("Show seasonality analysis", value=True)
        
        # Get trend data
        with st.spinner("🔍 Analyzing trends..."):
            trend_data = kpi_analyzer.analyze_trends(value_col, date_col, aggregation)
        
        if trend_data is None or trend_data.empty:
            st.warning("No trend data available. Please check your date and value columns.")
            return
        
        # Convert to DataFrame for Plotly
        trend_df = trend_data.reset_index()
        if len(trend_df.columns) == 2:
            trend_df.columns = [date_col, value_col]
        
        # Enhanced trend visualization with forecast
        fig_trend = go.Figure()
        
        # Actual trend line
        fig_trend.add_trace(go.Scatter(
            x=trend_df.iloc[:, 0],
            y=trend_df.iloc[:, 1],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add forecast if requested
        if show_forecast and len(trend_data) >= 6:
            with st.spinner("📊 Generating forecast..."):
                forecast_data = kpi_analyzer.generate_forecast(trend_data, periods=3)
            if forecast_data is not None and not forecast_data.empty:
                fig_trend.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    hovertemplate='<b>%{x}</b><br>Forecast: %{y:.2f}<extra></extra>'
                ))
        
        fig_trend.update_layout(
            title=f"{value_col} Trend Analysis ({aggregation})",
            xaxis_title="Date",
            yaxis_title=value_col,
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Enhanced Growth Analysis
        st.write("#### 📊 Growth Analysis")
        with st.spinner("Calculating growth rates..."):
            growth_rates = kpi_analyzer.calculate_comprehensive_growth_rates(trend_data, value_col)
        
        if growth_rates:
            cols = st.columns(5)
            growth_metrics = [
                ('MTD Growth', 'mtd_growth'),
                ('Monthly Growth', 'monthly_growth'),
                ('Quarterly Growth', 'quarterly_growth'),
                ('YTD Growth', 'ytd_growth'),
                ('Yearly Growth', 'yearly_growth')
            ]
            
            for col, (label, key) in zip(cols, growth_metrics):
                with col:
                    if key in growth_rates:
                        value = growth_rates[key]
                        delta_color = "normal" if value > 0 else "inverse"
                        st.metric(
                            label, 
                            f"{value:+.2f}%", 
                            delta_color=delta_color,
                            help=f"{label} rate"
                        )
                    else:
                        st.metric(label, "N/A", help="Insufficient data")
        
        # Enhanced Seasonality Analysis
        if show_seasonality:
            st.write("#### 🔄 Seasonality Analysis")
            with st.spinner("Analyzing seasonality..."):
                seasonality = kpi_analyzer.analyze_seasonality(trend_data, value_col)
            
            if seasonality is not None and not seasonality.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    seasonality_df = seasonality.reset_index()
                    if len(seasonality_df.columns) >= 2:
                        seasonality_df.columns = ['month', value_col]
                        
                        fig_seasonal = px.bar(
                            seasonality_df, x='month', y=value_col,
                            title="Monthly Seasonal Patterns",
                            color=value_col,
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_seasonal, use_container_width=True)
                
                with col2:
                    st.write("**Seasonal Insights**")
                    seasonal_insights = kpi_analyzer.generate_seasonal_insights(seasonality)
                    if seasonal_insights:
                        for insight in seasonal_insights:
                            st.write(f"• {insight}")
                    else:
                        st.info("No strong seasonal patterns detected")
            
            # Trend Statistics
            st.write("#### 📋 Trend Statistics")
            with st.spinner("Calculating trend statistics..."):
                trend_stats = kpi_analyzer.calculate_trend_statistics(trend_data)
            
            if trend_stats:
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    strength = trend_stats.get('trend_strength', 0)
                    st.metric("Trend Strength", f"{strength:.2f}", 
                             help="R-squared of linear trend (0-1)")
                with stat_cols[1]:
                    vol = trend_stats.get('volatility', 0)
                    st.metric("Volatility", f"{vol:.2f}%", 
                             help="Coefficient of variation")
                with stat_cols[2]:
                    best = trend_stats.get('best_month', 'N/A')
                    st.metric("Best Period", best)
                with stat_cols[3]:
                    worst = trend_stats.get('worst_month', 'N/A')
                    st.metric("Worst Period", worst)
                
    except Exception as e:
        st.error(f"Error in trend analysis: {str(e)}")

def render_enhanced_performance_metrics(kpi_analyzer):
    """Render enhanced performance metrics with advanced analytics"""
    
    with st.expander("💡 **Performance Metrics Guide**", expanded=False):
        st.markdown("""
        **Understanding Performance Metrics:**
        - **Segment Analysis**: Compare different groups (products, regions, etc.)
        - **Pareto Analysis**: Identify your top 20% that drive 80% of results
        - **Performance Tiers**: Categorize into top/middle/bottom performers
        - **Customer Analytics**: Understand customer behavior patterns
        
        **How to use:**
        1. Select segment column (category to analyze)
        2. Choose performance metric
        3. View segment comparisons
        4. Identify improvement opportunities
        """)
    
    st.subheader("🎯 Advanced Performance Metrics")
    
    df = kpi_analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        st.info("ℹ️ No categorical columns found for segment analysis.")
        # Fall back to distribution analysis only
        render_performance_distribution_only(kpi_analyzer, numeric_cols)
        return
    
    if not numeric_cols:
        st.info("ℹ️ No numeric columns found for performance analysis.")
        return
    
    # Segment analysis
    st.write("#### 🏷️ Segment Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        segment_col = st.selectbox(
            "Select segment column", 
            categorical_cols,
            key="segment_col"
        )
    
    with col2:
        metric_col = st.selectbox(
            "Select performance metric", 
            numeric_cols,
            key="segment_metric"
        )
    
    if not segment_col or not metric_col:
        st.info("👆 Please select both segment and metric columns")
        return
    
    try:
        # Enhanced segment performance with statistical significance
        with st.spinner("Analyzing segment performance..."):
            segment_performance = kpi_analyzer.analyze_segment_performance_with_stats(metric_col, segment_col)
        
        if segment_performance.empty:
            st.warning("No segment performance data available.")
            return
        
        # Convert to DataFrame for display
        segment_df = segment_performance.reset_index()
        
        # Ensure we have proper column names
        if len(segment_df.columns) >= 2:
            # Rename columns for clarity
            new_columns = ['Segment']
            stat_columns = ['Mean', 'Sum', 'Count', 'Std', 'Z-Score', 'P-Value', 'Significance']
            
            for i, col in enumerate(segment_df.columns[1:], 1):
                if i-1 < len(stat_columns):
                    new_columns.append(stat_columns[i-1])
                else:
                    new_columns.append(f'Stat_{i}')
            
            segment_df.columns = new_columns[:len(segment_df.columns)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Enhanced bar chart - use only available columns
                display_df = segment_df.head(10)  # Top 10 segments
                
                if 'Segment' in display_df.columns and 'Mean' in display_df.columns:
                    fig_segment = px.bar(
                        display_df,
                        x='Segment',
                        y='Mean',
                        title=f"Top 10 Segments by Average {metric_col}",
                        color='Mean',
                        color_continuous_scale='viridis',
                        hover_data=[col for col in ['Count', 'Std', 'Significance'] if col in display_df.columns]
                    )
                    fig_segment.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_segment, use_container_width=True)
                else:
                    st.warning("Insufficient data for segment visualization")
            
            with col2:
                st.write("**Segment Performance Summary**")
                
                # Performance summary cards - safely access columns
                if len(segment_df) > 0:
                    if 'Mean' in segment_df.columns and 'Count' in segment_df.columns:
                        top_segment = segment_df.iloc[0]
                        bottom_segment = segment_df.iloc[-1]
                        
                        top_segment_name = top_segment.get('Segment', 'Top Segment') 
                        bottom_segment_name = bottom_segment.get('Segment', 'Bottom Segment')
                        
                        st.metric(
                            f"🏆 Best: {top_segment_name}",
                            f"${top_segment['Mean']:,.2f}" if pd.notna(top_segment['Mean']) else "N/A",
                            f"{top_segment['Count']} records" if pd.notna(top_segment['Count']) else ""
                        )
                        
                        st.metric(
                            f"📉 Needs Improvement: {bottom_segment_name}",
                            f"${bottom_segment['Mean']:,.2f}" if pd.notna(bottom_segment['Mean']) else "N/A",
                            f"{bottom_segment['Count']} records" if pd.notna(bottom_segment['Count']) else "",
                            delta_color="inverse"
                        )
                        
                        # Performance gap
                        if (pd.notna(top_segment['Mean']) and pd.notna(bottom_segment['Mean']) 
                            and bottom_segment['Mean'] > 0):
                            gap_ratio = top_segment['Mean'] / bottom_segment['Mean']
                            st.metric(
                                "Performance Gap Ratio",
                                f"{gap_ratio:.1f}x",
                                "Top vs Bottom"
                            )
                    else:
                        st.warning("Missing required columns for performance summary")
            
            # Enhanced Pareto analysis
            st.write("#### 📊 Pareto Analysis (80/20 Rule)")
            with st.spinner("Performing Pareto analysis..."):
                pareto_data = kpi_analyzer.pareto_analysis(metric_col, segment_col)
            
            if pareto_data is not None and not pareto_data.empty:
                # Check if required columns exist in pareto_data
                if all(col in pareto_data.columns for col in ['segment', 'value', 'cumulative_percentage']):
                    fig_pareto = go.Figure()
                    
                    # Bar chart for values
                    fig_pareto.add_trace(go.Bar(
                        x=pareto_data['segment'],
                        y=pareto_data['value'],
                        name='Value',
                        marker_color='blue',
                        opacity=0.7
                    ))
                    
                    # Line for cumulative percentage
                    fig_pareto.add_trace(go.Scatter(
                        x=pareto_data['segment'],
                        y=pareto_data['cumulative_percentage'],
                        name='Cumulative %',
                        yaxis='y2',
                        line=dict(color='red', width=3),
                        marker=dict(color='red', size=6)
                    ))
                    
                    # 80% line
                    fig_pareto.add_hline(
                        y=80, line_dash="dash", 
                        line_color="green", 
                        annotation_text="80% Line",
                        yref="y2"
                    )
                    
                    fig_pareto.update_layout(
                        title="Pareto Analysis - Value Concentration",
                        xaxis_title=segment_col,
                        yaxis_title=metric_col,
                        yaxis2=dict(
                            title='Cumulative Percentage',
                            overlaying='y',
                            side='right',
                            range=[0, 100]
                        ),
                        showlegend=True,
                        xaxis_tickangle=-45
                    )
                    
                    st.plotly_chart(fig_pareto, use_container_width=True)
                    
                    # Pareto insights
                    eighty_percent_data = pareto_data[pareto_data['cumulative_percentage'] >= 80]
                    if not eighty_percent_data.empty:
                        eighty_percent_index = eighty_percent_data.index[0]
                        segments_for_80 = eighty_percent_index + 1
                        total_segments = len(pareto_data)
                        
                        st.info(f"""
                        **Pareto Insight**: {segments_for_80} out of {total_segments} segments ({(segments_for_80/total_segments)*100:.1f}%) 
                        contribute to 80% of total {metric_col}. Focus on these top segments for maximum impact.
                        """)
                else:
                    st.warning("Pareto analysis data format is incorrect")
    
    except Exception as e:
        st.error(f"Error in performance metrics: {str(e)}")
    
    # Advanced Performance Distribution Analysis
    render_performance_distribution_only(kpi_analyzer, numeric_cols)

def render_performance_distribution_only(kpi_analyzer, numeric_cols):
    """Render performance distribution when no categorical columns available"""
    st.write("#### 📊 Performance Distribution Analysis")
    
    if not numeric_cols:
        return
        
    performance_col = st.selectbox(
        "Select performance column for distribution", 
        numeric_cols, 
        key="perf_dist"
    )
    
    if not performance_col:
        return
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance tiers with enhanced visualization
            with st.spinner("Calculating performance tiers..."):
                tiers = kpi_analyzer.performance_tiers(performance_col, tiers=5)
            
            if tiers:
                # Enhanced tier analysis
                fig_tiers = px.pie(
                    names=list(tiers.keys()),
                    values=list(tiers.values()),
                    title="Performance Tier Distribution",
                    color=list(tiers.keys()),
                    color_discrete_sequence=px.colors.sequential.Viridis
                )
                st.plotly_chart(fig_tiers, use_container_width=True)
        
        with col2:
            st.write("**Performance Tier Analysis**")
            df = kpi_analyzer.df
            total_records = len(df)
            
            if tiers:
                for tier, count in tiers.items():
                    percentage = (count / total_records) * 100
                    st.progress(
                        min(int(percentage), 100),
                        text=f"{tier} Tier: {count} records ({percentage:.1f}%)"
                    )
                
                # Tier insights
                top_tier_pct = (tiers.get('Top', 0) / total_records) * 100
                bottom_tier_pct = (tiers.get('Bottom', 0) / total_records) * 100
                
                if top_tier_pct < 20:
                    st.success(f"**Opportunity**: Only {top_tier_pct:.1f}% in Top tier. Potential to move more segments up.")
                if bottom_tier_pct > 30:
                    st.warning(f"**Alert**: {bottom_tier_pct:.1f}% in Bottom tier. Need improvement strategies.")
            else:
                st.info("No performance tiers available for analysis")
        
        # Distribution visualization
        st.write("#### 📈 Performance Distribution")
        try:
            fig_dist = px.histogram(
                kpi_analyzer.df, x=performance_col,
                title=f"Distribution of {performance_col}",
                nbins=50,
                color_discrete_sequence=['#00CC96']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating distribution chart: {str(e)}")
            
    except Exception as e:
        st.error(f"Error in distribution analysis: {str(e)}")

def render_enhanced_comparative_analysis(kpi_analyzer):
    """Render enhanced comparative analysis with benchmarking"""
    
    with st.expander("💡 **Comparative Analysis Guide**", expanded=False):
        st.markdown("""
        **Understanding Comparative Analysis:**
        - **Benchmarking**: Compare segments against each other
        - **Performance Gaps**: Identify where you're under/over performing
        - **Statistical Testing**: Determine if differences are significant
        - **Goal Tracking**: Measure against targets
        
        **How to use:**
        1. Select benchmark column (groups to compare)
        2. Choose metric to compare
        3. Analyze performance differences
        4. Identify statistical significance
        """)
    
    st.subheader("📊 Advanced Comparative Analysis")
    
    df = kpi_analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols or not numeric_cols:
        st.info("ℹ️ Need both categorical and numeric columns for comparative analysis")
        return
    
    st.write("#### 🎯 Benchmarking Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        benchmark_col = st.selectbox(
            "Select benchmark column", 
            categorical_cols, 
            key="benchmark"
        )
    
    with col2:
        metric_col = st.selectbox(
            "Select metric for comparison", 
            numeric_cols, 
            key="bench_metric"
        )
    
    if not benchmark_col or not metric_col:
        st.info("👆 Please select both benchmark and metric columns")
        return
    
    try:
        # Enhanced benchmarking with statistical significance
        with st.spinner("Performing benchmark analysis..."):
            benchmarks = kpi_analyzer.benchmark_analysis_with_stats(metric_col, benchmark_col)
        
        if benchmarks is None or benchmarks.empty:
            st.warning("No benchmark data available.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced box plot
            fig_benchmark = px.box(
                df, x=benchmark_col, y=metric_col,
                title=f"{metric_col} Distribution by {benchmark_col}",
                color=benchmark_col
            )
            st.plotly_chart(fig_benchmark, use_container_width=True)
        
        with col2:
            # Benchmark summary
            st.write("**Benchmark Summary**")
            
            # Best and worst performers
            try:
                best_benchmark = benchmarks.loc[benchmarks['mean'].idxmax()]
                worst_benchmark = benchmarks.loc[benchmarks['mean'].idxmin()]
                
                st.metric(
                    f"🥇 Best: {best_benchmark.name}",
                    f"${best_benchmark['mean']:,.2f}",
                    f"±${best_benchmark.get('std', 0):,.2f}"
                )
                
                st.metric(
                    f"📊 Average",
                    f"${df[metric_col].mean():,.2f}",
                    "Overall mean"
                )
                
                st.metric(
                    f"🥉 Worst: {worst_benchmark.name}",
                    f"${worst_benchmark['mean']:,.2f}",
                    f"±${worst_benchmark.get('std', 0):,.2f}",
                    delta_color="inverse"
                )
                
                # Statistical significance
                if hasattr(benchmarks, 'attrs') and 'groups_significant' in benchmarks.attrs:
                    if benchmarks.attrs['groups_significant']:
                        st.success("✅ Groups show statistically significant differences")
                    else:
                        st.info("ℹ️ No statistically significant differences between groups")
                        
            except Exception as e:
                st.error(f"Error in benchmark summary: {str(e)}")
        
        # Enhanced Performance Gaps Analysis
        st.write("#### 📈 Performance Gaps Analysis")
        with st.spinner("Calculating performance gaps..."):
            performance_gaps = kpi_analyzer.calculate_performance_gaps(metric_col, benchmark_col)
        
        if performance_gaps:
            # Convert to DataFrame for better display
            gap_df = pd.DataFrame(performance_gaps)
            
            # Add color coding for gaps
            def color_gap(val):
                if val > 10:
                    return 'color: green; font-weight: bold'
                elif val < -10:
                    return 'color: red; font-weight: bold'
                else:
                    return 'color: orange'
            
            # Display styled dataframe
            try:
                styled_gap_df = gap_df.style.applymap(
                    color_gap, 
                    subset=['gap_percentage']
                ).format({
                    'segment_mean': '${:,.2f}',
                    'overall_mean': '${:,.2f}',
                    'gap_percentage': '{:+.1f}%'
                })
                
                st.dataframe(styled_gap_df, use_container_width=True)
                
                # Gap insights
                if not gap_df.empty:
                    largest_gap = max(gap_df['gap_percentage'], key=abs)
                    largest_gap_row = gap_df[gap_df['gap_percentage'] == largest_gap].iloc[0]
                    
                    if largest_gap > 20:
                        st.success(f"**Opportunity**: {largest_gap_row['segment']} outperforms by {largest_gap:.1f}%. Consider best practice replication.")
                    elif largest_gap < -20:
                        st.error(f"**Priority**: {largest_gap_row['segment']} underperforms by {abs(largest_gap):.1f}%. Immediate improvement needed.")
            except Exception as e:
                st.error(f"Error displaying performance gaps: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in comparative analysis: {str(e)}")
    
    # Advanced Goal Tracking
    st.write("#### 🎯 Goal Tracking & Performance vs Target")
    
    if numeric_cols:
        goal_col = st.selectbox(
            "Select metric for goal tracking", 
            numeric_cols, 
            key="goal_metric"
        )
        
        if goal_col:
            col1, col2 = st.columns(2)
            
            with col1:
                current_mean = df[goal_col].mean()
                target_value = st.number_input(
                    f"Set target for {goal_col}",
                    value=float(current_mean * 1.1),  # 10% above average
                    step=float(current_mean * 0.1) if current_mean > 0 else 1.0,
                    format="%.2f"
                )
            
            with col2:
                st.write("**Goal Achievement**")
                current_value = current_mean
                achievement_pct = (current_value / target_value) * 100 if target_value > 0 else 0
                
                st.metric(
                    "Goal Achievement",
                    f"{achievement_pct:.1f}%",
                    f"Target: ${target_value:,.2f}"
                )
                
                # Progress bar
                progress_value = min(int(achievement_pct), 100)
                st.progress(
                    progress_value,
                    text=f"Progress: {achievement_pct:.1f}% of target"
                )
                
                if achievement_pct >= 100:
                    st.success("🎉 Target achieved!")
                elif achievement_pct >= 80:
                    st.warning("📈 Close to target - keep going!")
                else:
                    st.info("📊 More effort needed to reach target")

def render_driver_analysis(kpi_analyzer):
    """Render comprehensive driver analysis"""
    
    with st.expander("💡 **Driver Analysis Guide**", expanded=False):
        st.markdown("""
        **Understanding Driver Analysis:**
        - **Correlation Analysis**: Find what factors influence your key metrics
        - **Impact Assessment**: Measure how much each factor matters
        - **Relationship Types**: Identify positive/negative relationships
        - **Actionable Insights**: Focus on factors you can control
        
        **How to use:**
        1. Select your target outcome metric
        2. Choose potential driver variables
        3. Analyze correlation strength and direction
        4. Identify key drivers for focus
        """)
    
    st.subheader("🔍 Driver Analysis - What Drives Your Results?")
    
    df = kpi_analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.info("ℹ️ Need at least 2 numeric columns for driver analysis")
        return
    
    st.write("#### 📈 Identify Key Business Drivers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        outcome_col = st.selectbox(
            "Select outcome variable (what you want to explain)", 
            numeric_cols, 
            key="outcome"
        )
    
    with col2:
        driver_cols = st.multiselect(
            "Select potential driver variables", 
            [col for col in numeric_cols if col != outcome_col],
            default=[col for col in numeric_cols if col != outcome_col][:min(3, len(numeric_cols)-1)]  # First 3 as default
        )
    
    if not outcome_col or not driver_cols:
        st.info("👆 Please select outcome and driver variables")
        return
    
    try:
        # Enhanced driver analysis
        with st.spinner("Analyzing drivers..."):
            driver_analysis = kpi_analyzer.comprehensive_driver_analysis(outcome_col, driver_cols)
        
        if not driver_analysis or 'correlations' not in driver_analysis:
            st.warning("No driver analysis results available.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Correlation matrix heatmap
            st.write("**Correlation Matrix**")
            analysis_cols = driver_cols + [outcome_col]
            try:
                corr_matrix = df[analysis_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Correlation Heatmap",
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating correlation matrix: {str(e)}")
        
        with col2:
            st.write("**Driver Impact Ranking**")
            
            # Sort drivers by absolute correlation
            sorted_drivers = sorted(
                driver_analysis['correlations'].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for i, (driver, correlation) in enumerate(sorted_drivers, 1):
                # Determine impact level and color
                abs_corr = abs(correlation)
                if abs_corr > 0.7:
                    impact = "🔥 High Impact"
                    color = "green"
                elif abs_corr > 0.4:
                    impact = "🟡 Medium Impact"
                    color = "orange"
                else:
                    impact = "🔵 Low Impact"
                    color = "blue"
                
                # Determine relationship type
                relationship = "Positive ↗️" if correlation > 0 else "Negative ↘️"
                
                st.metric(
                    label=f"{i}. {driver}",
                    value=f"{correlation:.3f}",
                    delta=f"{impact} | {relationship}",
                    delta_color="normal" if correlation > 0 else "inverse"
                )
        
        # Statistical Significance
        st.write("#### 📊 Statistical Significance")
        
        if driver_analysis.get('significant_drivers'):
            st.success(f"**✅ {len(driver_analysis['significant_drivers'])} Statistically Significant Drivers Found**")
            significant_df = pd.DataFrame(driver_analysis['significant_drivers'])
            
            # Format for better display
            display_cols = ['driver', 'correlation', 'p_value', 'strength']
            available_cols = [col for col in display_cols if col in significant_df.columns]
            
            if available_cols:
                st.dataframe(significant_df[available_cols].round(4), use_container_width=True)
        else:
            st.warning("⚠️ No statistically significant drivers found at p < 0.05 level")
        
        # Actionable Insights
        st.write("#### 💡 Actionable Driver Insights")
        
        top_driver = sorted_drivers[0] if sorted_drivers else None
        if top_driver:
            driver_name, driver_corr = top_driver
            
            if abs(driver_corr) > 0.6:
                st.success(f"""
                **🎯 Primary Driver Identified**: `{driver_name}`
                
                **📋 Recommendation**: Focus your efforts on improving {driver_name} as it has the strongest relationship with {outcome_col}.
                {"" if driver_corr > 0 else "⚠️ Note: This is a negative relationship - lower values might be better."}
                """)
            
            # Multiple driver insights
            strong_drivers = [d for d in sorted_drivers if abs(d[1]) > 0.5]
            if len(strong_drivers) >= 2:
                st.info(f"""
                **🔍 Multiple Strong Drivers**: You have {len(strong_drivers)} factors strongly correlated with {outcome_col}.
                Consider a multi-faceted strategy addressing these key areas simultaneously.
                """)
            
            # Weak drivers insight
            weak_drivers = [d for d in sorted_drivers if abs(d[1]) < 0.3]
            if weak_drivers:
                st.warning(f"""
                **📉 Weak Relationships**: {len(weak_drivers)} drivers show weak correlation with {outcome_col}.
                Consider whether these should be included in your analysis or if there are better metrics to track.
                """)
                
    except Exception as e:
        st.error(f"Error in driver analysis: {str(e)}")

def render_enhanced_strategic_insights(kpi_analyzer):
    """Render enhanced strategic insights with AI-powered recommendations"""
    
    with st.expander("💡 **Strategic Insights Guide**", expanded=False):
        st.markdown("""
        **Understanding Strategic Insights:**
        - **AI-Powered Analysis**: Automated insights from your data patterns
        - **Risk Identification**: Potential problems and opportunities
        - **Actionable Recommendations**: Specific steps to improve
        - **Impact Assessment**: Expected results from each action
        
        **How to use:**
        1. Click 'Generate Strategic Insights'
        2. Review automated findings
        3. Check recommendations
        4. Export insights for planning
        """)
    
    st.subheader("💡 Advanced Strategic Insights & AI Recommendations")
    
    # Insight configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        insight_depth = st.selectbox(
            "Analysis Depth",
            ["Quick Scan", "Standard Analysis", "Deep Dive"],
            index=1,
            key="insight_depth"
        )
    
    with col2:
        focus_area = st.selectbox(
            "Focus Area",
            ["All Areas", "Growth", "Efficiency", "Risk", "Customer Experience"],
            key="focus_area"
        )
    
    with col3:
        time_horizon = st.selectbox(
            "Time Horizon",
            ["Short-term (0-3 months)", "Medium-term (3-12 months)", "Long-term (1+ years)"],
            key="time_horizon"
        )
    
    if st.button("🚀 Generate Comprehensive Strategic Insights", type="primary", use_container_width=True):
        with st.spinner("🤖 AI is analyzing your data and generating strategic insights..."):
            try:
                # Comprehensive insights generation
                insights = kpi_analyzer.generate_comprehensive_strategic_insights(
                    depth=insight_depth,
                    focus_area=focus_area,
                    time_horizon=time_horizon
                )
                
                if not insights:
                    st.error("Failed to generate strategic insights. Please check your data.")
                    return
                
                st.success("✅ Strategic insights generated successfully!")
                
                # Display insights in an organized manner
                insight_categories = {
                    'performance_insights': '🎯 Performance Insights',
                    'growth_opportunities': '📈 Growth Opportunities', 
                    'risk_alerts': '⚠️ Risk Alerts',
                    'efficiency_insights': '⚡ Efficiency Insights',
                    'customer_insights': '👥 Customer Insights',
                    'competitive_insights': '🏆 Competitive Insights'
                }
                
                for category, display_name in insight_categories.items():
                    category_insights = insights.get(category, [])
                    if category_insights:  # Only show if there are insights
                        st.write(f"#### {display_name}")
                        
                        if isinstance(category_insights, list):
                            for insight in category_insights:
                                if category == 'risk_alerts':
                                    st.error(f"• {insight}")
                                elif category == 'growth_opportunities':
                                    st.success(f"• {insight}")
                                else:
                                    st.info(f"• {insight}")
                        else:
                            st.write(category_insights)
                        
                        st.markdown("---")
                
                # Enhanced Actionable Recommendations
                st.write("#### 🎯 AI-Powered Actionable Recommendations")
                recommendations = kpi_analyzer.generate_prioritized_recommendations()
                
                if not recommendations:
                    st.info("No specific recommendations generated. The data appears to be in good shape!")
                    return
                
                for i, recommendation in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{i}. {recommendation['area']}**")
                            st.write(f"   {recommendation['recommendation']}")
                            
                            # Progress tracker for implementation
                            implementation_key = f"imp_{i}_{hash(recommendation['area'])}"
                            if st.checkbox(f"Mark as implemented", key=implementation_key):
                                st.success("✅ Implemented")
                        
                        with col2:
                            impact_level = recommendation['impact'].split(' - ')[0]
                            impact_color = {
                                'High': 'green',
                                'Medium': 'orange', 
                                'Low': 'blue'
                            }.get(impact_level, 'gray')
                            
                            st.metric(
                                "Expected Impact",
                                impact_level,
                                recommendation['impact'].split(' - ')[1] if ' - ' in recommendation['impact'] else ""
                            )
                        
                        st.write("")
                
                # Export capabilities
                st.write("#### 📤 Export Insights")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📄 Export to PDF", use_container_width=True):
                        st.info("PDF export feature would be implemented here")
                
                with col2:
                    if st.button("📊 Export to Presentation", use_container_width=True):
                        st.info("Presentation export feature would be implemented here")
                
                with col3:
                    if st.button("📋 Copy Insights", use_container_width=True):
                        st.info("Insights copied to clipboard (simulated)")
                        
            except Exception as e:
                st.error(f"Error generating strategic insights: {str(e)}")

def render_executive_summary(kpi_analyzer):
    """Render executive summary dashboard"""
    
    with st.expander("💡 **Executive Summary Guide**", expanded=False):
        st.markdown("""
        **Understanding Executive Summary:**
        - **At-a-Glance View**: Quick overview of business health
        - **Key Highlights**: Most important findings
        - **Priority Actions**: What needs immediate attention
        - **Performance Scorecard**: Overall business scoring
        
        **How to use:**
        1. Review overall business health score
        2. Check priority actions
        3. View key metrics highlights
        4. Share with leadership team
        """)
    
    st.subheader("📋 Executive Summary Dashboard")
    
    # Generate executive summary
    with st.spinner("Generating executive summary..."):
        try:
            executive_data = kpi_analyzer.generate_executive_summary()
            
            if not executive_data:
                st.error("Failed to generate executive summary. Please check your data.")
                return
            
            # Overall Business Health Score
            st.write("#### 🏆 Overall Business Health Score")
            
            health_score = executive_data.get('health_score', 0)
            health_color = "green" if health_score >= 80 else "orange" if health_score >= 60 else "red"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Business Health Score",
                    f"{health_score}/100",
                    executive_data.get('health_trend', 'Stable')
                )
            
            with col2:
                st.metric(
                    "Growth Momentum",
                    executive_data.get('growth_momentum', 'Neutral'),
                    "Trend direction"
                )
            
            with col3:
                st.metric(
                    "Risk Level",
                    executive_data.get('risk_level', 'Medium'),
                    "Overall risk assessment"
                )
            
            # Health score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Business Health Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': health_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Key Highlights
            st.write("#### 🌟 Key Highlights")
            
            highlights = executive_data.get('key_highlights', [])
            if highlights:
                for highlight in highlights:
                    st.success(f"✅ {highlight}")
            else:
                st.info("No specific highlights generated.")
            
            # Priority Actions
            st.write("#### 🚨 Priority Actions")
            
            priorities = executive_data.get('priority_actions', [])
            if priorities:
                for i, priority in enumerate(priorities, 1):
                    with st.expander(f"Priority {i}: {priority['action']}", expanded=i==1):
                        st.write(f"**Impact**: {priority['impact']}")
                        st.write(f"**Timeline**: {priority['timeline']}")
                        st.write(f"**Owner**: {priority['owner']}")
            else:
                st.info("No priority actions identified.")
            
            # Performance Scorecard
            st.write("#### 📊 Performance Scorecard")
            
            scorecard = executive_data.get('performance_scorecard', {})
            if scorecard:
                score_cols = st.columns(len(scorecard))
                
                for col, (metric, score_info) in zip(score_cols, scorecard.items()):
                    with col:
                        score_value = score_info.get('score', 0)
                        st.metric(
                            metric,
                            f"{score_value}/10",
                            score_info.get('trend', '')
                        )
                        st.caption(score_info.get('description', ''))
            
            # Download Executive Summary
            st.download_button(
                label="📥 Download Executive Summary",
                data=str(executive_data),
                file_name="executive_summary_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error generating executive summary: {str(e)}")