import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application"""
    
    # OpenAI API Key
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    
    # App configuration
    APP_TITLE = "GenAI Data Analyst"
    APP_DESCRIPTION = "AI-Powered Data Analysis Platform"
    APP_VERSION = "1.0.0"
    
    # File upload settings
    MAX_FILE_SIZE = 200  # MB
    ALLOWED_FILE_TYPES = [
        'csv', 'xlsx', 'xls', 'json', 'parquet'
    ]
    
    # Analysis settings
    MAX_ROWS_DISPLAY = 1000
    AUTO_CLEAN_THRESHOLD = 0.5  # Auto-clean if missing values < 50%
    
    # Visualization settings
    CHART_HEIGHT = 400
    COLOR_PALETTE = 'viridis'
    
    @classmethod
    def setup_page_config(cls):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=cls.APP_TITLE,
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @classmethod
    def validate_api_key(cls):
        """Validate if OpenAI API key is available"""
        if not cls.OPENAI_API_KEY:
            st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables or .env file")
            return False
        return True

# Initialize configuration
config = Config()