import streamlit as st
from config import Config
from components.sidebar import render_sidebar
from components.welcome import render_welcome
from components.data_overview import render_data_overview
from components.data_cleaning import render_data_cleaning as render_data_cleaning
from components.eda_analysis import render_eda_analysis
from components.statistical_analysis import render_statistical_analysis
from components.business_insights import render_business_insights
from components.multi_file_analysis import render_multi_file_analysis
from components.ai_assistant import render_ai_assistant
from components.report_generator import render_professional_report_generator as render_report_generator
from utils.data_processor import DataProcessor

def main():
    """Main application function"""
    # Setup page configuration
    Config.setup_page_config()
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar
    current_page = render_sidebar()
    
    # Render main content based on selected page
    render_main_content(current_page)

def init_session_state():
    """Initialize session state variables"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'ai_assistant_history' not in st.session_state:
        st.session_state.ai_assistant_history = []

def render_main_content(current_page):
    """Render main content based on current page"""
    
    # Display header
    st.title(f"📊 {Config.APP_TITLE}")
    st.markdown("---")
    
    # Page routing
    if current_page == "Welcome":
        render_welcome()
    
    elif current_page == "Data Overview":
        if validate_data_uploaded():
            render_data_overview()
        else:
            st.warning("Please upload data files first from the Welcome page")
    
    elif current_page == "Data Cleaning":
        if validate_data_uploaded():
            render_data_cleaning()
        else:
            st.warning("Please upload data files first from the Welcome page")
    
    elif current_page == "EDA Analysis":
        if validate_data_uploaded():
            render_eda_analysis()
        else:
            st.warning("Please upload data files first from the Welcome page")
    
    elif current_page == "Statistical Analysis":
        if validate_data_uploaded():
            render_statistical_analysis()
        else:
            st.warning("Please upload data files first from the Welcome page")
    
    elif current_page == "Business Insights":
        if validate_data_uploaded():
            render_business_insights()
        else:
            st.warning("Please upload data files first from the Welcome page")
    
    elif current_page == "Multi-File Analysis":
        if validate_multiple_files():
            render_multi_file_analysis()
        else:
            st.warning("Please upload multiple data files first from the Welcome page")
    
    elif current_page == "AI Assistant":
        if Config.validate_api_key():
            render_ai_assistant()
        else:
            st.warning("OpenAI API key required for AI Assistant")
    
    elif current_page == "Report Generator":
        if validate_data_uploaded():
            render_report_generator()
        else:
            st.warning("Please upload data files first from the Welcome page")

def validate_data_uploaded():
    """Check if data has been uploaded"""
    return len(st.session_state.uploaded_files) > 0

def validate_multiple_files():
    """Check if multiple files have been uploaded"""
    return len(st.session_state.uploaded_files) > 1

if __name__ == "__main__":
    main()