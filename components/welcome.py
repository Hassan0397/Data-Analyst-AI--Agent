import streamlit as st
import pandas as pd
from utils.data_processor import DataProcessor
import base64
from typing import List, Dict
import time

def render_gradient_text(text: str, size: int = 40, weight: str = "bold") -> str:
    """Create gradient text effect"""
    return f"""
    <h1 style='
        font-size: {size}px;
        font-weight: {weight};
        background: linear-gradient(45deg, #0066FF, #00CCFF, #0066FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        text-align: left;
        line-height: 1.1;
    '>{text}</h1>
    """

def render_feature_card(icon: str, title: str, description: str):
    """Render professional feature card with perfect spacing"""
    return f"""
    <div class="feature-card" style="
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 28px 24px;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        margin-bottom: 24px;
    ">
        <div>
            <div style="font-size: 2.5rem; margin-bottom: 16px; color: #0066FF; text-align: center;">{icon}</div>
            <h3 style="color: #1F2937; margin-bottom: 12px; font-size: 1.1rem; font-weight: 600; line-height: 1.3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; text-align: center;">{title}</h3>
        </div>
        <p style="color: #6B7280; font-size: 0.9rem; line-height: 1.5; margin: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; text-align: center;">{description}</p>
    </div>
    """

def render_stats_card(icon: str, value: str, label: str):
    """Render consistent stats card with perfect spacing"""
    return f"""
    <div class='stats-card' style="
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 0px;
        transition: all 0.3s ease;
        min-width: 140px;
    ">
        <div style='font-size: 1.8rem; margin-bottom: 8px; color: #0066FF;'>{icon}</div>
        <div style='font-size: 1.3rem; font-weight: 800; color: #1F2937; margin-bottom: 4px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1; letter-spacing: -0.5px;'>{value}</div>
        <div style='color: #6B7280; font-size: 0.75rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.2; text-transform: uppercase; letter-spacing: 0.5px;'>{label}</div>
    </div>
    """

def render_step_card(step: str, title: str, description: str):
    """Render consistent step card with perfect spacing"""
    return f"""
    <div style='
        text-align: center;
        background: white;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 32px 24px;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 0px;
    '>
        <div class='step-indicator' style='
            background: linear-gradient(45deg, #0066FF, #00CCFF);
            border-radius: 50%;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px auto;
            font-weight: 700;
            color: white;
            font-size: 1.1rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        '>{step}</div>
        <h3 style='color: #1F2937; margin-bottom: 12px; font-size: 1.1rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>{title}</h3>
        <p style='color: #6B7280; font-size: 0.9rem; line-height: 1.5; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>{description}</p>
    </div>
    """

def render_welcome():
    """Render premium welcome page with perfect spacing and alignment"""
    
    # Custom CSS for perfect spacing and consistency
    st.markdown("""
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');
    
    .main {
        background: white;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .stApp {
        background: white;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Feature cards spacing */
    .feature-card {
        margin-bottom: 24px !important;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 24px -4px rgba(0, 0, 0, 0.12), 0 6px 8px -2px rgba(0, 0, 0, 0.08);
        border: 1px solid #0066FF !important;
    }
    
    .stats-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -4px rgba(0, 0, 0, 0.15);
        border: 1px solid #0066FF !important;
    }
    
    /* Perfect upload area spacing */
    .upload-area {
        background: white !important;
        border: 2px dashed #D1D5DB !important;
        border-radius: 20px;
        padding: 60px 40px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin: 0 auto;
    }
    
    .upload-area:hover {
        border: 2px dashed #0066FF !important;
        background: #F8FAFF !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 102, 255, 0.12);
    }
    
    /* File preview perfect spacing */
    .file-preview {
        background: #F9FAFB;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        border-left: 4px solid #0066FF;
        border: 1px solid #E5E7EB;
        min-height: 70px;
        display: flex;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .file-preview:hover {
        background: #F3F4F6;
        transform: translateX(4px);
    }
    
    .trusted-badge {
        background: linear-gradient(45deg, #0066FF, #00CCFF);
        padding: 16px 32px;
        border-radius: 25px;
        display: inline-block;
        color: white;
        font-weight: 800;
        font-size: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 8px 24px rgba(0, 102, 255, 0.4);
        letter-spacing: 1px;
        text-align: center;
        margin: 0 auto;
    }
    
    /* Perfect section dividers */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #E5E7EB, transparent);
        margin: 60px 0;
    }
    
    .section-title {
        color: #1F2937;
        text-align: center;
        margin-bottom: 48px;
        font-size: 2.1rem;
        font-weight: 700;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.2;
    }
    
    .section-subtitle {
        color: #6B7280;
        text-align: center;
        margin-bottom: 40px;
        font-size: 1.15rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        line-height: 1.5;
    }
    
    /* Perfect column spacing */
    .row-spacing {
        margin-bottom: 8px;
    }
    
    /* Stats grid perfect spacing */
    .stats-grid {
        gap: 16px;
        margin-bottom: 20px;
    }
    
    /* Ensure all text uses consistent font */
    .stMarkdown, .stText, .stTitle, .stHeader {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }
    
    /* Button perfect styling */
    .browse-button {
        background: linear-gradient(45deg, #0066FF, #00CCFF);
        padding: 16px 36px;
        border-radius: 25px;
        display: inline-block;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.3);
        margin-top: 20px;
    }
    
    .browse-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 102, 255, 0.4);
    }
    
    /* Hero section specific styles */
    .hero-left {
        padding-right: 40px;
    }
    
    .hero-right {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 100%;
    }
    
    .hero-subtitle {
        color: #6B7280;
        font-size: 1.25rem;
        line-height: 1.6;
        margin-bottom: 40px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Stats card specific improvements */
    .stats-column {
        min-width: 160px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with Perfect Spacing
    st.markdown('<div class="row-spacing"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="hero-left">', unsafe_allow_html=True)
        st.markdown(render_gradient_text("🚀 GenAI Data Analyst Pro"), unsafe_allow_html=True)
        st.markdown("""
        <div class='hero-subtitle'>
        Enterprise-grade AI-powered data analysis platform trusted by Fortune 500 companies
        </div>
        """, unsafe_allow_html=True)
        
        # Key Metrics - Perfect Grid Spacing
        st.markdown('<div class="stats-grid">', unsafe_allow_html=True)
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.markdown(render_stats_card("📊", "10M+", "Datasets Analyzed"), unsafe_allow_html=True)
            
        with metrics_col2:
            st.markdown(render_stats_card("🏢", "5K+", "Enterprise Clients"), unsafe_allow_html=True)
            
        with metrics_col3:
            st.markdown(render_stats_card("⚡", "99.9%", "Accuracy Rate"), unsafe_allow_html=True)
            
        with metrics_col4:
            st.markdown(render_stats_card("🕒", "24/7", "AI Support"), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="hero-right">', unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; padding: 20px; height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;'>
            <div style='font-size: 8rem; margin-bottom: 32px; color: #0066FF; text-align: center;'>📈</div>
            <div class='trusted-badge'>
                ENTERPRISE READY
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Features Grid with Perfect Spacing
    st.markdown('<h2 class="section-title">🛠️ Enterprise-Grade Features</h2>', unsafe_allow_html=True)
    
    features_col1, features_col2, features_col3 = st.columns(3)
    
    with features_col1:
        st.markdown(render_feature_card("🤖", "AI-Powered Insights", "Advanced machine learning algorithms uncover hidden patterns and predictive insights"), unsafe_allow_html=True)
        st.markdown(render_feature_card("🔗", "Multi-Source Integration", "Connect and analyze data from databases, APIs, cloud storage, and local files"), unsafe_allow_html=True)
        st.markdown(render_feature_card("📊", "Real-Time Dashboards", "Interactive dashboards with live data updates and collaborative features"), unsafe_allow_html=True)
    
    with features_col2:
        st.markdown(render_feature_card("⚡", "Automated ETL", "Intelligent data pipelines with automated cleaning, transformation, and validation"), unsafe_allow_html=True)
        st.markdown(render_feature_card("🔒", "Enterprise Security", "SOC 2 compliant with end-to-end encryption and role-based access control"), unsafe_allow_html=True)
        st.markdown(render_feature_card("📈", "Advanced Analytics", "Statistical modeling, time series forecasting, and sentiment analysis"), unsafe_allow_html=True)
    
    with features_col3:
        st.markdown(render_feature_card("🌐", "Cloud Native", "Scalable infrastructure with auto-scaling and global CDN support"), unsafe_allow_html=True)
        st.markdown(render_feature_card("🤝", "Team Collaboration", "Real-time collaboration, comments, and version control for teams"), unsafe_allow_html=True)
        st.markdown(render_feature_card("🚀", "API First", "RESTful APIs for seamless integration with your existing tools"), unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Enhanced File Upload Section with Perfect Spacing
    st.markdown('<h2 class="section-title">🚀 Start Your Analysis</h2>', unsafe_allow_html=True)
    
    # Upload Area with Perfect Centering
    uploaded_files = st.file_uploader(
        " ",
        type=['csv', 'xlsx', 'xls', 'json', 'parquet'],
        accept_multiple_files=True,
        help="Drag and drop your files here or click to browse",
        key="premium_uploader"
    )
    
    # Custom upload area with perfect spacing
    upload_container = st.container()
    with upload_container:
        if not uploaded_files:
            st.markdown("""
            <div class='upload-area'>
                <div style='font-size: 3.5rem; margin-bottom: 20px; color: #9CA3AF;'>📁</div>
                <h3 style='color: #1F2937; margin-bottom: 12px; font-size: 1.3rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Drag & Drop Your Files</h3>
                <p style='color: #6B7280; margin-bottom: 0; font-size: 0.95rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                    Supported formats: CSV, Excel, JSON, Parquet • Max 200MB per file
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Browse button with perfect spacing
            st.markdown("""
            <div style='text-align: center; margin-top: 24px;'>
                <div class='browse-button'>
                    BROWSE FILES
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if uploaded_files:
        process_uploaded_files(uploaded_files)
        
        # File Preview Section with Perfect Spacing
        st.markdown("""
        <div style='margin-top: 48px;'>
            <h3 style='color: #1F2937; margin-bottom: 24px; font-size: 1.3rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                📋 Uploaded Files Preview
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        for file in uploaded_files:
            st.markdown(f"""
            <div class='file-preview'>
                <div style='display: flex; justify-content: space-between; align-items: center; width: 100%;'>
                    <div>
                        <strong style='color: #1F2937; font-size: 1rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>{file.name}</strong>
                        <div style='color: #6B7280; font-size: 0.85rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
                            {file.size / 1024 / 1024:.2f} MB • {file.type}
                        </div>
                    </div>
                    <div style='color: #10B981; font-weight: 600; font-size: 0.9rem; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>✅ Ready</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Enhanced Quick Start Guide with Perfect Spacing
    st.markdown('<h2 class="section-title">🎯 Get Started in Minutes</h2>', unsafe_allow_html=True)
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown(render_step_card("1", "Upload Data", "Drag & drop your files or connect to databases, APIs, or cloud storage"), unsafe_allow_html=True)
    
    with guide_col2:
        st.markdown(render_step_card("2", "AI Analysis", "Our AI automatically cleans, analyzes, and generates insights"), unsafe_allow_html=True)
    
    with guide_col3:
        st.markdown(render_step_card("3", "Get Insights", "Export reports, share dashboards, or integrate via API"), unsafe_allow_html=True)
    
    # Trusted By Section with Perfect Spacing
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 60px 0;'>
        <h4 style='color: #6B7280; margin-bottom: 40px; text-transform: uppercase; letter-spacing: 2px; font-size: 0.85rem; font-weight: 600; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Trusted By Industry Leaders</h4>
        <div style='display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 48px;'>
            <div style='color: #1F2937; font-weight: 600; font-size: 1.1rem; opacity: 0.7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Google</div>
            <div style='color: #1F2937; font-weight: 600; font-size: 1.1rem; opacity: 0.7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Microsoft</div>
            <div style='color: #1F2937; font-weight: 600; font-size: 1.1rem; opacity: 0.7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Amazon</div>
            <div style='color: #1F2937; font-weight: 600; font-size: 1.1rem; opacity: 0.7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Tesla</div>
            <div style='color: #1F2937; font-weight: 600; font-size: 1.1rem; opacity: 0.7; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>Netflix</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def process_uploaded_files(uploaded_files):
    """Process uploaded files with enhanced UI feedback"""
    
    if uploaded_files and uploaded_files != st.session_state.get('uploaded_files', []):
        st.session_state.uploaded_files = uploaded_files
        st.session_state.processed_data = {}
        
        processor = DataProcessor()
        
        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"🔄 Processing {file.name}...")
            
            try:
                df = processor.load_data(file)
                if df is not None:
                    st.session_state.processed_data[file.name] = {
                        'dataframe': df,
                        'file_type': file.type,
                        'file_size': file.size,
                        'original_columns': df.columns.tolist(),
                        'shape': df.shape,
                        'memory_usage': df.memory_usage(deep=True).sum()
                    }
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Success message with details
                    st.success(f"""
                    ✅ **{file.name}** successfully processed
                    - **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns
                    - **Size:** {file.size / 1024 / 1024:.2f} MB
                    - **Status:** Ready for analysis
                    """)
                else:
                    st.error(f"❌ Failed to process: {file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {file.name}: {str(e)}")
        
        # Completion
        if st.session_state.processed_data:
            progress_bar.progress(100)
            status_text.text("🎉 All files processed successfully!")
            
            # Success celebration with metrics
            total_rows = sum(data['shape'][0] for data in st.session_state.processed_data.values())
            total_columns = sum(data['shape'][1] for data in st.session_state.processed_data.values())
            
            st.balloons()
            st.success(f"""
            ## 🚀 Analysis Ready!
            
            **{len(uploaded_files)} files** processed with **{total_rows:,} total rows** and **{total_columns} columns**
            
            ➡️ **Proceed to Data Overview** to begin your analysis
            """)
            
            # Set initial state
            first_file = list(st.session_state.processed_data.keys())[0]
            st.session_state.current_file = first_file
            st.session_state.analysis_complete = False