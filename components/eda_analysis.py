import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
from typing import Dict, List, Any, Tuple, Optional
import concurrent.futures
from utils.eda_analyzer import EnterpriseEDAAnalyzer

warnings.filterwarnings('ignore')

def render_eda_analysis():
    """Render Enterprise Exploratory Data Analysis interface"""
    
    st.header("🧠 Enterprise EDA with AI Insights")
    
    # Beginner's Guide
    with st.expander("🎯 **Beginner's Guide: How to Use This Tool**", expanded=True):
        st.markdown("""
        ### Welcome! Here's how to get the most from this analysis tool:
        
        **🚀 Quick Start for Non-Technical Users:**
        1. **Upload your data** - Excel, CSV files work best
        2. **Enable AI Insights** in sidebar for smart analysis
        3. **Start with Data Intelligence** tab to understand your data
        4. **Use AI Insights** for automatic business recommendations
        5. **Check Action Plan** for specific next steps
        
        **📊 What Each Tab Does:**
        - **Data Intelligence**: Overview of your data quality and structure
        - **AI Insights**: Smart analysis and business recommendations  
        - **Advanced Analytics**: Deep statistical analysis
        - **Business Impact**: Calculate ROI and business value
        - **Action Plan**: Step-by-step guidance on what to do next
        - **Traditional EDA**: Classic statistical analysis methods
        
        **💡 Pro Tips:**
        - The AI can understand your business context - describe what you're trying to achieve
        - Start with the Action Plan tab for guided analysis
        - Use the ROI calculator to justify investments in data projects
        """)
    
    if not st.session_state.get('processed_data'):
        st.warning("No data available. Please upload files first.")
        return
    
    selected_file = st.session_state.get('current_file', 'dataset')
    data_info = st.session_state.processed_data.get(selected_file, {})
    
    if not data_info or 'dataframe' not in data_info:
        st.error("No valid dataset found. Please upload data first.")
        return
        
    df = data_info['dataframe']
    
    # Enhanced AI Configuration
    with st.sidebar:
        st.subheader("🔧 AI Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", 
                               help="Optional: For AI-powered insights. Get one from https://platform.openai.com/",
                               key="openai_api_key")
        
        enable_ai = st.checkbox("Enable AI Insights", value=False, key="enable_ai")
        
        if enable_ai and not api_key:
            st.warning("⚠️ Please enter your OpenAI API key to enable AI insights")
        
        st.subheader("⚡ Advanced Options")
        confidence_threshold = st.slider("Confidence Threshold", 0.7, 0.95, 0.8)
        sample_size = st.number_input("Sample Size", min_value=1000, max_value=500000, value=10000)
        enable_parallel = st.checkbox("Enable Parallel Processing", value=True)
        
        st.subheader("🔍 Analysis Focus")
        business_context = st.selectbox(
            "Business Domain",
            ["General Analytics", "E-commerce", "Finance", "Healthcare", "Marketing", "Manufacturing", "Custom"]
        )
        
        # Data Quality Goals
        st.subheader("🎯 Analysis Goals")
        analysis_goals = st.multiselect(
            "What are you trying to achieve?",
            ["Improve Data Quality", "Find Business Insights", "Predict Trends", 
             "Identify Risks", "Optimize Operations", "Increase Revenue", "Reduce Costs"]
        )
    
    # Enhanced dataset optimization
    if len(df) > 10000:
        df = optimize_large_dataset_enhanced(df, enable_parallel)
    
    # FIXED: Properly pass API key to analyzer
    analyzer_api_key = api_key if enable_ai and api_key else None
    analyzer = EnterpriseEDAAnalyzer(df, analyzer_api_key)
    
    # Enhanced tabs with progress tracking
    eda_tabs = st.tabs([
        "📊 Data Intelligence", 
        "🤖 AI Insights", 
        "🔍 Advanced Analytics",
        "📈 Business Impact",
        "🚀 Action Plan",
        "🔧 Traditional EDA"
    ])
    
    # Progress tracking
    analysis_progress = st.progress(0)
    
    with eda_tabs[0]:
        analysis_progress.progress(10)
        render_enhanced_data_intelligence(analyzer)
    
    with eda_tabs[1]:
        analysis_progress.progress(30)
        render_enhanced_ai_insights(analyzer, enable_ai, api_key, business_context, analysis_goals)
    
    with eda_tabs[2]:
        analysis_progress.progress(50)
        render_enhanced_advanced_analytics(analyzer)
    
    with eda_tabs[3]:
        analysis_progress.progress(70)
        render_enhanced_business_impact(analyzer)
    
    with eda_tabs[4]:
        analysis_progress.progress(85)
        render_enhanced_action_plan(analyzer, analysis_goals)
    
    with eda_tabs[5]:
        analysis_progress.progress(100)
        render_enhanced_traditional_eda(analyzer)

def render_enhanced_data_intelligence(analyzer):
    """Render comprehensive data intelligence dashboard"""
    st.subheader("📊 Data Intelligence Dashboard")
    
    df = analyzer.df
    basic_stats = analyzer.get_basic_stats()
    data_quality_score = analyzer.calculate_data_quality_score()
    
    # Enhanced Key Metrics with quality scoring
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset Size", f"{basic_stats['dataset_shape'][0]:,} × {basic_stats['dataset_shape'][1]}", 
                 help="Number of rows and columns in your dataset")
    with col2:
        st.metric("Data Quality Score", f"{data_quality_score['overall_score']:.1f}/10",
                 delta=f"{data_quality_score['overall_score'] - 5:.1f}" if data_quality_score['overall_score'] > 5 else None)
    with col3:
        st.metric("Missing Values", f"{basic_stats['missing_values']:,}",
                 help="Cells with no data that might need attention")
    with col4:
        st.metric("Memory Usage", f"{basic_stats['memory_usage_mb']:.1f} MB")
    
    # Data Quality Score Breakdown
    st.write("#### 🎯 Data Quality Assessment")
    quality_cols = st.columns(5)
    quality_metrics = [
        ('Completeness', data_quality_score['completeness'], '#00C851'),
        ('Accuracy', data_quality_score['accuracy'], '#33b5e5'),
        ('Consistency', data_quality_score['consistency'], '#ffbb33'),
        ('Timeliness', data_quality_score['timeliness'], '#ff4444'),
        ('Validity', data_quality_score['validity'], '#aa66cc')
    ]
    
    for idx, (metric, score, color) in enumerate(quality_metrics):
        with quality_cols[idx]:
            st.metric(metric, f"{score:.1f}/10")
    
    # Enhanced Data Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 📋 Data Preview")
        st.info("First 10 rows of your data for quick inspection")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.write("#### 📝 Column Information")
        st.info("Detailed information about each column including data types and quality metrics")
        col_info = analyzer.get_enhanced_column_info()
        st.dataframe(col_info, use_container_width=True)
    
    # Enhanced Data Quality Heatmap
    st.write("#### 🔍 Missing Values Heatmap")
    missing_matrix = df.isnull().astype(int)
    fig_missing = px.imshow(
        missing_matrix.T,
        title="Missing Values Pattern Analysis",
        color_continuous_scale='Blues',
        aspect="auto"
    )
    st.plotly_chart(fig_missing, use_container_width=True)
    
    # Data Quality Recommendations
    if data_quality_score['overall_score'] < 7:
        st.warning("**⚠️ Data Quality Improvement Opportunities:**")
        recommendations = analyzer.generate_data_quality_recommendations()
        for rec in recommendations[:3]:
            st.write(f"• {rec}")

def render_enhanced_ai_insights(analyzer, enable_ai, api_key, business_context, analysis_goals):
    """Render enhanced AI-powered insights"""
    st.subheader("🤖 AI-Powered Analytical Insights")
    
    if not enable_ai or not api_key:
        st.info("""
        **🔒 AI Insights Locked** 
        
        Enable AI Insights in the sidebar to unlock:
        - **Smart Business Recommendations**: AI analyzes your data and suggests specific actions
        - **Risk Identification**: Automatic detection of data quality and business risks
        - **ROI Opportunities**: Find hidden revenue and cost-saving opportunities
        - **Strategic Roadmap**: Step-by-step plan for data-driven decision making
        
        *Get an API key from [OpenAI](https://platform.openai.com/) to enable these features*
        """)
        return
    
    # Enhanced context collection
    st.write("### 🎯 Tell Us About Your Business Goals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_context = st.text_area(
            "**Business Context & Objectives**",
            placeholder="Example: This is e-commerce transaction data from our online store. We want to improve customer conversion rates and reduce shopping cart abandonment...",
            height=120,
            help="Describe your business, what this data represents, and what you're trying to achieve"
        )
        
        insight_depth = st.selectbox(
            "**Analysis Depth**",
            ["Quick Scan", "Comprehensive Analysis", "Deep Dive"],
            help="Quick Scan: High-level insights | Comprehensive: Balanced detail | Deep Dive: Maximum detail"
        )
    
    with col2:
        focus_area = st.selectbox(
            "**Primary Focus Area**",
            ["Overall Analysis", "Data Quality", "Business Impact", "Risk Analysis", 
             "Growth Opportunities", "Operational Efficiency", "Customer Insights"],
            help="What aspect of the data should we focus on?"
        )
        
        # Enhanced industry context
        industry_specifics = st.text_input(
            "**Industry Specific Context**",
            placeholder="E.g., We're in SaaS with monthly subscriptions, or We're a retail chain with seasonal patterns...",
            help="Any industry-specific context that would help the AI understand your data better"
        )
    
    # Goals integration
    goals_text = ""
    if analysis_goals:
        goals_text = f"Primary Goals: {', '.join(analysis_goals)}"
    
    full_context = f"""
    Business Domain: {business_context}
    Focus Area: {focus_area}
    Analysis Depth: {insight_depth}
    {goals_text}
    Industry Context: {industry_specifics}
    User Description: {analysis_context}
    """
    
    if st.button("🧠 Generate AI Analysis", type="primary", use_container_width=True):
        with st.spinner("🤖 AI Analyst processing your data and generating strategic insights..."):
            try:
                # Check if analyzer has valid client
                if not analyzer.llm_client:
                    st.error("❌ AI client not configured. Please check your API key and ensure 'Enable AI Insights' is checked.")
                    return
                    
                insights = analyzer.generate_enhanced_ai_insights(full_context)
                
                if "error" not in insights:
                    display_enhanced_ai_insights(insights)
                    
                    # Add interactive features
                    with st.expander("💬 Provide Feedback on These Insights"):
                        feedback = st.selectbox(
                            "How helpful were these insights?",
                            ["Very Helpful", "Somewhat Helpful", "Not Helpful", "Need More Detail"]
                        )
                        if st.button("Submit Feedback"):
                            st.success("Thank you! Your feedback helps improve the AI.")
                            
                else:
                    st.error(f"AI Analysis failed: {insights['error']}")
                    if "API key" in insights['error'] or "invalid" in insights['error'].lower():
                        st.info("🔑 Please check your OpenAI API key is correct and has sufficient credits.")
                    else:
                        st.info("💡 Please try again or check your internet connection.")
                    
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.info("Please try again or check your internet connection")

def display_enhanced_ai_insights(insights):
    """Display structured enhanced AI insights"""
    
    st.success("✅ AI Analysis Complete! Here's Your Strategic Assessment")
    
    # Executive Summary
    if 'executive_summary' in insights:
        st.subheader("🎯 Executive Summary")
        st.info(insights['executive_summary'])
    
    # Key Findings with visual indicators
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Business Opportunities")
        opportunities = insights.get('business_opportunities', ['No specific opportunities identified'])
        for i, opportunity in enumerate(opportunities[:4], 1):
            with st.container():
                st.success(f"**{i}. {opportunity}**")
        
        st.subheader("⚠️ Critical Risks")
        risks = insights.get('critical_risks', ['No major risks identified'])
        for i, risk in enumerate(risks[:3], 1):
            with st.container():
                st.error(f"**{i}. {risk}**")
    
    with col2:
        st.subheader("💡 Strategic Recommendations")
        recommendations = insights.get('strategic_recommendations', ['No specific recommendations'])
        for i, recommendation in enumerate(recommendations[:4], 1):
            with st.container():
                st.info(f"**{i}. {recommendation}**")
        
        st.subheader("💰 Quick Wins & ROI")
        quick_wins = insights.get('quick_wins', ['No specific quick wins identified'])
        for i, win in enumerate(quick_wins[:3], 1):
            with st.container():
                st.success(f"**🏆 {win}**")
    
    # Enhanced Analytics Roadmap
    st.subheader("🛠️ Data & Analytics Roadmap")
    roadmap = insights.get('analytics_roadmap', ['No specific roadmap provided'])
    
    timeline_cols = st.columns(3)
    phases = {
        "Phase 1: Immediate (1-4 weeks)": roadmap[:2],
        "Phase 2: Short-term (1-3 months)": roadmap[2:4],
        "Phase 3: Strategic (3-6 months)": roadmap[4:6]
    }
    
    for idx, (phase_name, phase_items) in enumerate(phases.items()):
        with timeline_cols[idx]:
            st.write(f"**{phase_name}**")
            for item in phase_items:
                if item:
                    st.write(f"• {item}")
    
    # Data Quality Insights
    if 'data_quality_insights' in insights:
        st.subheader("🔍 Data Quality Assessment")
        quality_insights = insights['data_quality_insights']
        for insight in quality_insights[:3]:
            st.write(f"• {insight}")

def render_enhanced_advanced_analytics(analyzer):
    """Render enhanced advanced analytical visualizations"""
    st.subheader("🔬 Advanced Analytical Methods")
    
    # Beginner Guide for Advanced Analytics
    with st.expander("📚 Understanding Advanced Analytics"):
        st.markdown("""
        **What These Advanced Techniques Can Tell You:**
        
        - **PCA (Dimensionality Analysis)**: Finds the most important patterns in your data
        - **Feature Importance**: Shows which factors matter most for predictions
        - **Clustering**: Groups similar data points together automatically
        - **Anomaly Detection**: Finds unusual patterns that might indicate problems or opportunities
        
        **No Math Required!** The AI does the complex calculations - you get the insights.
        """)
    
    df = analyzer.df
    numeric_cols = analyzer.numeric_cols
    
    if len(numeric_cols) < 2:
        st.info("""
        **ℹ️ Need More Numeric Data**
        
        Advanced analytics works best with numeric data (numbers). 
        Your dataset needs at least 2 numeric columns for these analyses.
        
        *Tip: If you have categories, consider encoding them as numbers for analysis*
        """)
        return
    
    # Enhanced Dimensionality Reduction with incremental PCA
    st.write("#### 📊 Dimensionality Analysis (PCA)")
    
    with st.spinner("Performing advanced dimensionality analysis..."):
        advanced_results = analyzer.perform_enhanced_advanced_analysis()
    
    if 'pca' in advanced_results:
        col1, col2 = st.columns(2)
        
        with col1:
            # PCA Variance Explained
            fig_pca = px.line(
                x=range(1, len(advanced_results['pca']['explained_variance_ratio']) + 1),
                y=advanced_results['pca']['cumulative_variance'],
                title="PCA - Cumulative Variance Explained",
                labels={'x': 'Principal Components', 'y': 'Cumulative Variance'}
            )
            fig_pca.add_hline(y=0.8, line_dash="dash", line_color="red", 
                             annotation_text="80% Variance Threshold")
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            # Component Importance
            components_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(advanced_results['pca']['explained_variance_ratio']))],
                'Variance Explained': advanced_results['pca']['explained_variance_ratio'],
                'Cumulative': advanced_results['pca']['cumulative_variance']
            })
            st.dataframe(components_df, use_container_width=True)
        
        # Component Interpretation
        st.write("**🔍 What These Components Mean:**")
        component_insights = analyzer.interpret_pca_components(advanced_results['pca'])
        for insight in component_insights[:3]:
            st.write(f"• {insight}")
    
    # Enhanced Feature Importance Analysis
    st.write("#### 🎯 Feature Importance Analysis")
    
    if len(numeric_cols) >= 2:
        # Enhanced target selection with guidance
        target_col = st.selectbox(
            "Select target variable to predict", 
            numeric_cols, 
            key="target_var",
            help="Choose the column you want to understand or predict better"
        )
        feature_cols = [col for col in numeric_cols if col != target_col]
        
        if feature_cols:
            with st.spinner("Calculating feature importance using multiple methods..."):
                importance_results = analyzer.calculate_feature_importance(target_col, feature_cols)
                
                # Display multiple importance methods
                method_tabs = st.tabs([method for method in importance_results.keys()])
                
                for idx, (method_name, importance_df) in enumerate(importance_results.items()):
                    with method_tabs[idx]:
                        fig_importance = px.bar(
                            importance_df, 
                            x='importance', 
                            y='feature',
                            title=f"{method_name} - Feature Importance for {target_col}",
                            orientation='h'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Interpretation
                        top_features = importance_df.nlargest(3, 'importance')
                        st.write(f"**Top 3 Most Important Features ({method_name}):**")
                        for _, row in top_features.iterrows():
                            st.write(f"• **{row['feature']}** (importance: {row['importance']:.3f})")
    
    # Enhanced Clustering Analysis
    st.write("#### 👥 Smart Clustering Analysis")
    
    if 'clustering' in advanced_results:
        cluster_info = advanced_results['clustering']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clusters Identified", len(cluster_info['cluster_sizes']))
        with col2:
            st.metric("Clustering Quality", f"{cluster_info['silhouette_score']:.3f}")
        with col3:
            st.metric("Largest Cluster", max(cluster_info['cluster_sizes'].values()))
        
        # Cluster interpretation
        st.write("**🧩 Cluster Characteristics:**")
        cluster_insights = analyzer.interpret_clusters(cluster_info)
        for insight in cluster_insights[:4]:
            st.write(f"• {insight}")
    
    # Enhanced Anomaly Detection
    st.write("#### 🔍 Advanced Anomaly Detection")
    
    if 'anomaly_detection' in advanced_results:
        anomaly_info = advanced_results['anomaly_detection']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomalies Detected", 
                     f"{len(anomaly_info['anomaly_indices'])} ({anomaly_info['anomaly_percentage']:.1f}%)")
        with col2:
            st.metric("Risk Level", 
                     "High" if anomaly_info['anomaly_percentage'] > 5 else "Medium" if anomaly_info['anomaly_percentage'] > 2 else "Low")
        
        if anomaly_info['anomaly_indices']:
            st.write("**📋 Sample Anomalies Found:**")
            anomaly_sample = df.iloc[anomaly_info['anomaly_indices'][:5]]
            st.dataframe(anomaly_sample, use_container_width=True)
            
            # Anomaly interpretation
            st.write("**💡 What These Anomalies Might Mean:**")
            anomaly_insights = analyzer.interpret_anomalies(anomaly_sample)
            for insight in anomaly_insights[:3]:
                st.write(f"• {insight}")

def render_enhanced_business_impact(analyzer):
    """Render enhanced business impact analysis"""
    st.subheader("💰 Business Impact Assessment")
    
    # Beginner Guide for Business Impact
    with st.expander("💡 How to Use This ROI Calculator"):
        st.markdown("""
        **Calculate the Business Value of Data Analysis:**
        
        1. **Fill in your current business metrics** (conversion rates, order values, etc.)
        2. **Estimate potential improvements** based on data insights
        3. **See the financial impact** automatically calculated
        4. **Use these numbers to justify** data projects to management
        
        *Even rough estimates can show significant ROI!*
        """)
    
    # Enhanced ROI Calculator
    st.write("#### 📈 ROI Impact Calculator")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Current Metrics")
        current_conversion = st.number_input("Current Conversion Rate (%)", 
                                           min_value=0.1, max_value=100.0, value=2.0,
                                           help="What percentage of visitors currently convert?")
        monthly_visitors = st.number_input("Monthly Visitors", 
                                         min_value=100, max_value=10000000, value=10000,
                                         help="How many people visit your store/service monthly?")
    
    with col2:
        st.subheader("Improvement Potential")
        improved_conversion = st.number_input("Expected Conversion Improvement (%)", 
                                            min_value=0.1, max_value=50.0, value=0.5,
                                            help="Realistic improvement from data insights")
        avg_order_value = st.number_input("Average Order Value ($)", 
                                        min_value=1, max_value=10000, value=100,
                                        help="How much does the average customer spend?")
    
    with col3:
        st.subheader("Costs & Efficiency")
        implementation_cost = st.number_input("Implementation Cost ($)", 
                                           min_value=0, max_value=100000, value=5000,
                                           help="One-time cost to implement changes")
        analysis_time_saved = st.number_input("Analysis Time Saved (hours/month)", 
                                           min_value=0, max_value=160, value=20,
                                           help="How many hours will this save your team monthly?")
        hourly_rate = st.number_input("Average Hourly Rate ($)", 
                                    min_value=10, max_value=500, value=100,
                                    help="Cost of analyst/manager time")
    
    # Calculate Enhanced ROI
    current_revenue = monthly_visitors * (current_conversion/100) * avg_order_value
    new_revenue = monthly_visitors * ((current_conversion + improved_conversion)/100) * avg_order_value
    revenue_increase = new_revenue - current_revenue
    time_savings_value = analysis_time_saved * hourly_rate
    
    monthly_roi = revenue_increase + time_savings_value
    annual_roi = monthly_roi * 12
    payback_period = implementation_cost / monthly_roi if monthly_roi > 0 else float('inf')
    roi_percentage = (annual_roi / implementation_cost) * 100 if implementation_cost > 0 else float('inf')
    
    # Enhanced ROI Metrics
    st.write("#### 💵 Financial Impact Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Monthly Revenue Impact", f"${revenue_increase:,.0f}", 
                 delta=f"+{improved_conversion}% conversion")
    with col2:
        st.metric("Monthly Efficiency Savings", f"${time_savings_value:,.0f}",
                 delta=f"{analysis_time_saved} hours saved")
    with col3:
        st.metric("Annual ROI", f"${annual_roi:,.0f}",
                 delta=f"{roi_percentage:.0f}% ROI" if roi_percentage != float('inf') else "N/A")
    with col4:
        st.metric("Payback Period", 
                 f"{payback_period:.1f} months" if payback_period != float('inf') else "N/A",
                 delta="Quick ROI" if payback_period < 6 else "Long-term")
    
    # Enhanced ROI Visualization
    if monthly_roi > 0:
        roi_data = pd.DataFrame({
            'Category': ['Revenue Increase', 'Time Savings'],
            'Value': [revenue_increase, time_savings_value],
            'Type': ['Revenue', 'Efficiency']
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_roi = px.pie(roi_data, values='Value', names='Category', 
                             title="ROI Composition",
                             color='Type',
                             color_discrete_map={'Revenue': '#00C851', 'Efficiency': '#33b5e5'})
            st.plotly_chart(fig_roi, use_container_width=True)
        
        with col2:
            # Monthly projection
            months = list(range(13))
            cumulative_roi = [monthly_roi * month - implementation_cost for month in months]
            
            fig_projection = px.area(
                x=months, y=cumulative_roi,
                title="Cumulative ROI Projection",
                labels={'x': 'Months', 'y': 'Cumulative ROI ($)'}
            )
            fig_projection.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_projection, use_container_width=True)
    
    # Business Case Template
    with st.expander("📋 Generate Business Case Document"):
        st.write("**Use these numbers to build your business case:**")
        business_case = f"""
        **Data Analytics Investment Business Case**
        
        **Current State:**
        - Monthly Visitors: {monthly_visitors:,}
        - Conversion Rate: {current_conversion}%
        - Average Order Value: ${avg_order_value:,.0f}
        - Monthly Revenue: ${current_revenue:,.0f}
        
        **Proposed Improvement:**
        - Conversion Rate Improvement: +{improved_conversion}%
        - New Conversion Rate: {current_conversion + improved_conversion}%
        - Expected Monthly Revenue Increase: ${revenue_increase:,.0f}
        - Monthly Efficiency Savings: ${time_savings_value:,.0f}
        
        **Financial Impact:**
        - Implementation Cost: ${implementation_cost:,.0f}
        - Monthly ROI: ${monthly_roi:,.0f}
        - Annual ROI: ${annual_roi:,.0f}
        - Payback Period: {payback_period:.1f} months
        
        **Recommendation:**
        This analysis demonstrates strong financial returns with a payback period of {payback_period:.1f} months and annual ROI of ${annual_roi:,.0f}.
        """
        st.code(business_case)

def render_enhanced_action_plan(analyzer, analysis_goals):
    """Render enhanced automated action plan"""
    st.subheader("🚀 Data-Driven Action Plan")
    
    # Beginner Guide for Action Plan
    with st.expander("🎯 How to Use This Action Plan"):
        st.markdown("""
        **Your Step-by-Step Guide to Data Success:**
        
        - **Priority Levels**: 
          - 🔴 HIGH: Critical issues that need immediate attention
          - 🟡 MEDIUM: Important improvements for the near term  
          - 🟢 LOW: Enhancements for future optimization
        
        - **Each action includes**: What to do, why it matters, who should do it, and timeline
        - **Track progress**: Check off completed items to see your progress
        
        *Start with HIGH priority items for maximum impact!*
        """)
    
    if st.button("📋 Generate Smart Action Plan", type="primary", use_container_width=True):
        with st.spinner("Creating personalized strategic action plan based on your data and goals..."):
            # Enhanced analysis with goals integration
            basic_stats = analyzer.get_basic_stats()
            distributions = analyzer.analyze_distributions()
            correlations = analyzer.get_strong_correlations(threshold=0.7)
            data_quality_score = analyzer.calculate_data_quality_score()
            
            action_plan = []
            
            # Enhanced Data Quality Actions
            if basic_stats['missing_values'] > 0:
                urgency = "HIGH" if data_quality_score['completeness'] < 6 else "MEDIUM"
                action_plan.append({
                    'priority': urgency,
                    'action': 'Implement data quality pipeline',
                    'description': f"Address {basic_stats['missing_values']} missing values affecting data reliability",
                    'timeline': '2 weeks',
                    'owner': 'Data Engineering',
                    'impact': 'High',
                    'effort': 'Medium',
                    'prerequisites': 'Data source access'
                })
            
            # Goal-specific actions
            if "Improve Data Quality" in analysis_goals:
                action_plan.append({
                    'priority': 'HIGH',
                    'action': 'Establish data quality monitoring',
                    'description': 'Set up automated checks and alerts for data quality issues',
                    'timeline': '3 weeks',
                    'owner': 'Data Governance',
                    'impact': 'High',
                    'effort': 'Medium',
                    'prerequisites': 'Data quality framework'
                })
            
            if "Find Business Insights" in analysis_goals:
                action_plan.append({
                    'priority': 'MEDIUM',
                    'action': 'Create executive dashboard',
                    'description': 'Build interactive dashboards for key business metrics',
                    'timeline': '4 weeks',
                    'owner': 'BI Team',
                    'impact': 'High',
                    'effort': 'High',
                    'prerequisites': 'Clear KPIs defined'
                })
            
            # Analytical Actions
            skewed_columns = [col for col, stats in distributions.items() 
                            if 'skewness' in stats and abs(stats['skewness']) > 1]
            if skewed_columns:
                action_plan.append({
                    'priority': 'MEDIUM',
                    'action': 'Apply data transformations',
                    'description': f"Transform skewed variables for better analysis: {', '.join(skewed_columns[:3])}",
                    'timeline': '1 week',
                    'owner': 'Data Science',
                    'impact': 'Medium',
                    'effort': 'Low',
                    'prerequisites': 'Statistical understanding'
                })
            
            # Correlation-based actions
            if not correlations.empty:
                action_plan.append({
                    'priority': 'MEDIUM',
                    'action': 'Investigate strong correlations',
                    'description': f"Analyze {len(correlations)} strong variable relationships for business insights",
                    'timeline': '1 week',
                    'owner': 'Business Analytics',
                    'impact': 'High',
                    'effort': 'Medium',
                    'prerequisites': 'Business context'
                })
            
            # Enhanced standard business actions
            standard_actions = [
                {
                    'priority': 'HIGH',
                    'action': 'Validate key business metrics',
                    'description': 'Ensure data accuracy for executive reporting and decision making',
                    'timeline': '3 days',
                    'owner': 'Business Analytics',
                    'impact': 'Critical',
                    'effort': 'Low',
                    'prerequisites': 'Metric definitions'
                },
                {
                    'priority': 'MEDIUM',
                    'action': 'Create monitoring dashboards',
                    'description': 'Build real-time monitoring for key KPIs and business metrics',
                    'timeline': '4 weeks',
                    'owner': 'BI Team',
                    'impact': 'High',
                    'effort': 'High',
                    'prerequisites': 'Data infrastructure'
                },
                {
                    'priority': 'LOW',
                    'action': 'Document data lineage',
                    'description': 'Create comprehensive data documentation and lineage tracking',
                    'timeline': '2 weeks',
                    'owner': 'Data Governance',
                    'impact': 'Medium',
                    'effort': 'Medium',
                    'prerequisites': 'Data catalog'
                }
            ]
            
            action_plan.extend(standard_actions)
            
            # Display enhanced action plan
            st.success(f"✅ Generated {len(action_plan)} personalized actionable items")
            
            # Priority filtering
            priority_filter = st.multiselect(
                "Filter by Priority:",
                ["HIGH", "MEDIUM", "LOW"],
                default=["HIGH", "MEDIUM"]
            )
            
            filtered_actions = [action for action in action_plan if action['priority'] in priority_filter]
            
            # Progress tracking
            completed_actions = st.session_state.get('completed_actions', set())
            
            for i, action in enumerate(filtered_actions, 1):
                # Priority color coding
                priority_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
                
                with st.expander(f"{priority_icon[action['priority']]} {i}. {action['action']} ({action['priority']} Priority)", 
                               expanded=action['priority'] == 'HIGH'):
                    
                    # Completion checkbox
                    action_key = f"action_{i}"
                    is_completed = st.checkbox("Mark as completed", 
                                             value=action_key in completed_actions,
                                             key=action_key)
                    
                    if is_completed:
                        completed_actions.add(action_key)
                    elif action_key in completed_actions:
                        completed_actions.remove(action_key)
                    
                    st.session_state.completed_actions = completed_actions
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**📝 Description:** {action['description']}")
                        st.write(f"**⏰ Timeline:** {action['timeline']}")
                        st.write(f"**👤 Owner:** {action['owner']}")
                    
                    with col2:
                        st.write(f"**💼 Business Impact:** {action['impact']}")
                        st.write(f"**⚡ Effort Required:** {action['effort']}")
                        st.write(f"**📋 Prerequisites:** {action['prerequisites']}")
                    
                    # Action buttons
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"Start {action['action']}", key=f"btn_start_{i}"):
                            st.success(f"✅ {action['action']} initiated!")
                            st.info("Next: Assign resources and set up tracking in your project management tool")
                    with col2:
                        if st.button("Get Templates", key=f"btn_templates_{i}"):
                            st.info(f"📋 Downloading templates for {action['action']}...")
                    with col3:
                        if st.button("Need Help", key=f"btn_help_{i}"):
                            st.info(f"🆘 Connecting you with resources for {action['action']}...")
            
            # Progress summary
            total_actions = len(action_plan)
            completed_count = len(completed_actions)
            progress_percent = (completed_count / total_actions) * 100 if total_actions > 0 else 0
            
            st.write("### 📊 Action Plan Progress")
            progress_col1, progress_col2 = st.columns([3, 1])
            with progress_col1:
                st.progress(progress_percent / 100)
            with progress_col2:
                st.write(f"{completed_count}/{total_actions} completed")

def render_enhanced_traditional_eda(analyzer):
    """Render enhanced traditional EDA components"""
    st.subheader("🔧 Traditional EDA Analysis")
    
    # Enhanced beginner guide
    with st.expander("📖 Traditional EDA Guide for Beginners"):
        st.markdown("""
        **Classic Data Analysis Techniques Made Easy:**
        
        **Descriptive Stats**: Basic numbers about your data (averages, ranges, etc.)
        **Univariate Analysis**: Understanding one variable at a time
        **Bivariate Analysis**: Relationships between two variables  
        **Multivariate Analysis**: Complex relationships between multiple variables
        **Trends & Patterns**: How your data changes over time
        
        **💡 Tip**: Start with Descriptive Stats, then move to Univariate analysis to understand each column individually.
        """)
    
    # Reuse your existing EDA functions here with enhancements
    eda_tabs = st.tabs([
        "📊 Descriptive Statistics",
        "📈 Univariate Analysis", 
        "🔗 Bivariate Analysis",
        "🌐 Multivariate Analysis",
        "📅 Trends & Patterns"
    ])
    
    with eda_tabs[0]:
        render_enhanced_descriptive_stats(analyzer)
    
    with eda_tabs[1]:
        render_enhanced_univariate_analysis(analyzer)
    
    with eda_tabs[2]:
        render_enhanced_bivariate_analysis(analyzer)
    
    with eda_tabs[3]:
        render_enhanced_multivariate_analysis(analyzer)
    
    with eda_tabs[4]:
        render_enhanced_trends_patterns(analyzer)

def render_enhanced_descriptive_stats(analyzer):
    """Render enhanced descriptive statistics"""
    st.subheader("📊 Descriptive Statistics")
    
    df = analyzer.df
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        # Enhanced summary statistics
        st.write("#### 📋 Summary Statistics")
        st.info("Basic statistical measures for numeric columns")
        summary_stats = numeric_df.describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Additional statistics with enhanced insights
        st.write("#### 🔍 Advanced Statistics")
        additional_stats = pd.DataFrame({
            'Variance': numeric_df.var(),
            'Skewness': numeric_df.skew(),
            'Kurtosis': numeric_df.kurtosis(),
            'Coefficient of Variation': (numeric_df.std() / numeric_df.mean().replace(0, np.nan)) * 100
        })
        st.dataframe(additional_stats, use_container_width=True)
        
        # Statistical insights
        st.write("#### 💡 Statistical Insights")
        insights = analyzer.generate_statistical_insights()
        for insight in insights[:5]:
            st.write(f"• {insight}")
    
    # Enhanced categorical statistics
    categorical_df = df.select_dtypes(include=['object', 'category'])
    if not categorical_df.empty:
        st.write("#### 📝 Categorical Data Summary")
        for col in categorical_df.columns[:5]:
            with st.expander(f"**{col}** Analysis"):
                value_counts = df[col].value_counts()
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Unique values:** {df[col].nunique()}")
                    st.write(f"**Most frequent:** {value_counts.index[0] if len(value_counts) > 0 else 'N/A'}")
                    st.write(f"**Frequency of top value:** {value_counts.iloc[0] if len(value_counts) > 0 else 0}")
                with col2:
                    st.bar_chart(value_counts.head(10))

def render_enhanced_univariate_analysis(analyzer):
    """Render enhanced univariate analysis"""
    st.subheader("📈 Univariate Analysis")
    
    df = analyzer.df
    
    # Column selection
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Numeric Variables", "Categorical Variables"],
        horizontal=True
    )
    
    if analysis_type == "Numeric Variables":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(
                        df, x=selected_col, 
                        title=f"Distribution of {selected_col}",
                        nbins=50
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(
                        df, y=selected_col,
                        title=f"Box Plot of {selected_col}"
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Q-Q Plot
                st.write("#### Q-Q Plot (Normality Check)")
                fig_qq = analyzer.create_qq_plot(selected_col)
                if fig_qq:
                    st.plotly_chart(fig_qq, use_container_width=True)
        else:
            st.info("No numeric columns found for analysis")
    
    else:  # Categorical Variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            selected_col = st.selectbox("Select categorical column", categorical_cols)
            
            if selected_col:
                # Bar chart
                value_counts = df[selected_col].value_counts().head(20)
                fig_bar = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Frequency of {selected_col}",
                    labels={'x': selected_col, 'y': 'Count'}
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Pie chart for top categories
                if len(value_counts) <= 10:
                    fig_pie = px.pie(
                        names=value_counts.index,
                        values=value_counts.values,
                        title=f"Proportion of {selected_col}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No categorical columns found for analysis")

def render_enhanced_bivariate_analysis(analyzer):
    """Render enhanced bivariate analysis"""
    st.subheader("🔗 Bivariate Analysis")
    
    df = analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Numeric vs Numeric", "Numeric vs Categorical", "Categorical vs Categorical"],
        key="bivariate_type"
    )
    
    if analysis_type == "Numeric vs Numeric":
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis column", numeric_cols, key="num_x")
            with col2:
                y_col = st.selectbox("Y-axis column", numeric_cols, key="num_y")
            
            if x_col and y_col:
                # Scatter plot
                fig_scatter = px.scatter(
                    df, x=x_col, y=y_col,
                    title=f"{x_col} vs {y_col}",
                    trendline="ols"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Correlation
                correlation = df[x_col].corr(df[y_col])
                st.write(f"**Correlation coefficient:** {correlation:.3f}")
                
                if abs(correlation) > 0.7:
                    st.info("💡 Strong correlation detected")
                elif abs(correlation) > 0.3:
                    st.info("💡 Moderate correlation detected")
        else:
            st.info("Need at least 2 numeric columns for numeric vs numeric analysis")
    
    elif analysis_type == "Numeric vs Categorical":
        if numeric_cols and categorical_cols:
            num_col = st.selectbox("Numeric column", numeric_cols, key="num_cat_num")
            cat_col = st.selectbox("Categorical column", categorical_cols, key="num_cat_cat")
            
            if num_col and cat_col:
                # Box plots by category
                fig_box = px.box(
                    df, x=cat_col, y=num_col,
                    title=f"{num_col} by {cat_col}"
                )
                fig_box.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Violin plot
                fig_violin = px.violin(
                    df, x=cat_col, y=num_col,
                    title=f"Distribution of {num_col} by {cat_col}"
                )
                fig_violin.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.info("Need both numeric and categorical columns for this analysis")
    
    else:  # Categorical vs Categorical
        if len(categorical_cols) >= 2:
            cat1 = st.selectbox("First categorical column", categorical_cols, key="cat1")
            cat2 = st.selectbox("Second categorical column", categorical_cols, key="cat2")
            
            if cat1 and cat2:
                # Cross tabulation
                cross_tab = pd.crosstab(df[cat1], df[cat2])
                st.write("#### Cross Tabulation")
                st.dataframe(cross_tab, use_container_width=True)
                
                # Stacked bar chart
                fig_stacked = px.bar(
                    cross_tab,
                    title=f"{cat1} vs {cat2}",
                    barmode='stack'
                )
                st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.info("Need at least 2 categorical columns for categorical vs categorical analysis")

def render_enhanced_multivariate_analysis(analyzer):
    """Render enhanced multivariate analysis"""
    st.subheader("🌐 Multivariate Analysis")
    
    df = analyzer.df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) >= 3:
        # Correlation matrix
        st.write("#### Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix Heatmap",
            aspect="auto",
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Strong correlations
        st.write("#### Strong Correlations (|r| > 0.7)")
        strong_corrs = analyzer.get_strong_correlations(threshold=0.7)
        if not strong_corrs.empty:
            st.dataframe(strong_corrs, use_container_width=True)
        else:
            st.info("No strong correlations found")
        
        # 3D Scatter plot
        st.write("#### 3D Scatter Plot")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
        with col2:
            y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y")
        with col3:
            z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z")
        
        color_col = st.selectbox(
            "Color by", 
            [None] + numeric_cols + df.select_dtypes(include=['object', 'category']).columns.tolist(),
            key="3d_color"
        )
        
        if x_col and y_col and z_col:
            fig_3d = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=color_col if color_col else None,
                title=f"3D Scatter: {x_col} vs {y_col} vs {z_col}"
            )
            st.plotly_chart(fig_3d, use_container_width=True)
    
    else:
        st.info("Need at least 3 numeric columns for comprehensive multivariate analysis")

def render_enhanced_trends_patterns(analyzer):
    """Render enhanced trends and patterns analysis"""
    st.subheader("📈 Trends & Patterns Analysis")
    
    df = analyzer.df
    
    # Time series analysis
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    if date_cols:
        st.write("#### Time Series Analysis")
        date_col = st.selectbox("Select date column", date_cols, key="date_col")
        value_col = st.selectbox("Select value column", 
                               df.select_dtypes(include=[np.number]).columns.tolist(),
                               key="value_col")
        
        if date_col and value_col:
            # Time series plot
            ts_df = df.set_index(date_col).sort_index()
            fig_ts = px.line(
                ts_df, y=value_col,
                title=f"{value_col} Over Time"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Rolling average
            window = st.slider("Rolling average window", 1, 30, 7, key="rolling_window")
            ts_df[f'rolling_{window}'] = ts_df[value_col].rolling(window=window).mean()
            
            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=ts_df.index, y=ts_df[value_col],
                name='Original', line=dict(color='blue')
            ))
            fig_rolling.add_trace(go.Scatter(
                x=ts_df.index, y=ts_df[f'rolling_{window}'],
                name=f'Rolling {window} days', line=dict(color='red')
            ))
            fig_rolling.update_layout(title=f"Rolling Average ({window} days)")
            st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Seasonality analysis
    st.write("#### Seasonality Analysis")
    if date_cols and len(numeric_cols := df.select_dtypes(include=[np.number]).columns.tolist()) > 0:
        seasonality_col = st.selectbox("Column for seasonality analysis", numeric_cols, key="seasonality_col")
        if seasonality_col and date_col:
            # Extract time components
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            
            # Monthly patterns
            monthly_avg = df.groupby('month')[seasonality_col].mean()
            fig_monthly = px.line(
                x=monthly_avg.index, y=monthly_avg.values,
                title=f"Monthly Pattern for {seasonality_col}",
                labels={'x': 'Month', 'y': f'Average {seasonality_col}'}
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Anomaly detection
    st.write("#### Anomaly Detection")
    if st.button("Detect Anomalies", key="detect_anomalies"):
        anomalies = analyzer.detect_anomalies()
        if not anomalies.empty:
            st.write(f"**Detected {len(anomalies)} anomalies:**")
            st.dataframe(anomalies, use_container_width=True)
        else:
            st.info("No anomalies detected using IQR method")

@st.cache_data(show_spinner=False, max_entries=10)
def perform_cached_analysis(_analyzer, analysis_type):
    """Enhanced cached analytical operations with memory optimization"""
    if analysis_type == "distributions":
        return _analyzer.analyze_distributions()
    elif analysis_type == "correlations":
        return _analyzer.analyze_correlations()
    elif analysis_type == "advanced":
        return _analyzer.perform_enhanced_advanced_analysis()
    return {}

def optimize_large_dataset_enhanced(df, enable_parallel=True):
    """Enhanced optimization for enterprise-scale datasets"""
    
    # Memory usage before optimization
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    # Downcast numeric types with enhanced logic
    for col in df.select_dtypes(include=[np.number]):
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Integer columns
        if df[col].dtype in ['int64', 'int32']:
            if col_min >= 0:
                if col_max < 255:
                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
                elif col_max < 65535:
                    df[col] = pd.to_numeric(df[col], downcast='unsigned')
            else:
                if col_min > -128 and col_max < 127:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif col_min > -32768 and col_max < 32767:
                    df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Float columns
        elif df[col].dtype in ['float64', 'float32']:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Enhanced categorical conversion with parallel processing
    if enable_parallel and len(df) > 100000:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        def optimize_column(col):
            if len(df[col].unique()) / len(df) < 0.5:
                return col, df[col].astype('category')
            return col, df[col]
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(optimize_column, categorical_cols))
        
        for col, optimized_data in results:
            df[col] = optimized_data
    else:
        # Sequential processing for smaller datasets
        for col in df.select_dtypes(include=['object']):
            if len(df[col].unique()) / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    # Memory usage after optimization
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = initial_memory - final_memory
    
    if memory_saved > 10:
        st.sidebar.success(f"🎯 Optimized memory usage: {memory_saved:.1f} MB saved")
    
    return df