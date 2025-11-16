import streamlit as st
from config import Config

def render_sidebar():
    """Render the application sidebar"""
    
    with st.sidebar:
        st.title("🧭 Navigation")
        st.markdown("---")
        
        # Page selection
        page_options = [
            "Welcome",
            "Data Overview", 
            "Data Cleaning",
            "EDA Analysis",
            "Statistical Analysis",
            "Business Insights",
            "Multi-File Analysis",
            "AI Assistant",
            "Report Generator"
        ]
        
        selected_page = st.radio(
            "Select Page",
            page_options,
            index=0
        )
        
        st.markdown("---")
        
        # File information
        if st.session_state.uploaded_files:
            st.subheader("📁 Uploaded Files")
            for i, file in enumerate(st.session_state.uploaded_files):
                st.write(f"{i+1}. {file.name}")
        
        st.markdown("---")
        
        # API Status
        st.subheader("🔑 API Status")
        if Config.OPENAI_API_KEY:
            st.success("OpenAI API: Connected ✅")
        else:
            st.error("OpenAI API: Not Configured ❌")
            
        # App info
        st.markdown("---")
        st.markdown(f"**Version:** {Config.APP_VERSION}")
        st.markdown("Built with ❤️ using Streamlit")
    
    return selected_page