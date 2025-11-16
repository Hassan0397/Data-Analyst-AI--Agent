# ai_assistant.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import Config
import json
import re
import io
import base64
from datetime import datetime
import tempfile
import os
import sys

# OpenAI import with version compatibility
try:
    from openai import OpenAI
    OPENAI_NEW_VERSION = True
except ImportError:
    import openai
    OPENAI_NEW_VERSION = False

def render_ai_assistant():
    """Render AI Assistant interface"""
    
    st.header("🤖 AI Data Analysis Assistant Pro")
    
    if not Config.validate_api_key():
        st.error("OpenAI API key required for AI Assistant")
        st.info("Please add your OpenAI API key to the config.py file")
        return
    
    # Initialize OpenAI client with version compatibility
    client = initialize_openai_client()
    if not client:
        return
    
    # Initialize chat history
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = [
            {
                "role": "system", 
                "content": get_system_prompt()
            }
        ]
    
    # Enhanced sidebar for AI settings
    with st.sidebar:
        st.header("⚙️ AI Settings")
        
        # Model selection with availability check
        available_models = get_available_models()
        selected_model = st.selectbox(
            "AI Model",
            available_models,
            index=0,
            help="Choose the AI model for analysis. GPT-4 provides better analysis but may not be available in all accounts."
        )
        
        # Temperature control
        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Lower for more deterministic, higher for more creative"
        )
        
        # Max tokens
        max_tokens = st.slider(
            "Max Response Length",
            min_value=500,
            max_value=4000,
            value=2000,
            step=500
        )
        
        # Model info
        st.markdown("---")
        st.write("**Model Info:**")
        if "gpt-4" in selected_model:
            st.success("✅ GPT-4 - Best for complex analysis")
        else:
            st.info("🤖 GPT-3.5 Turbo - Fast and cost-effective")
        
        # Chat management
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.ai_messages = [
                    {"role": "system", "content": get_system_prompt()}
                ]
                st.rerun()
        with col2:
            if st.button("💾 Export Chat", use_container_width=True):
                export_chat_history()
        
        # API version info
        st.markdown("---")
        st.caption(f"OpenAI API: {'New Version' if OPENAI_NEW_VERSION else 'Legacy Version'}")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display chat messages
        display_chat_messages()
        
        # Check for quick actions before processing user input
        if hasattr(st.session_state, 'quick_action'):
            handle_quick_actions(client, selected_model, temperature, max_tokens)
        
        # Enhanced user input with options
        user_input = st.chat_input("Ask me anything about your data, SQL queries, analysis, etc...")
        
        if user_input:
            process_user_message(client, user_input, selected_model, temperature, max_tokens)
    
    with col2:
        # Quick actions panel
        display_quick_actions_panel()
        
        # Data context panel
        display_data_context_panel()
        
        # Code execution panel
        display_code_execution_panel()

def get_available_models():
    """Get list of available models based on common access patterns"""
    # Default models that should work for most users
    base_models = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]
    
    # Try to add GPT-4 models but they might not be available
    gpt4_models = [
        "gpt-4",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-0125-preview"
    ]
    
    # Start with base models that should always work
    available_models = base_models.copy()
    
    # Check if we've already determined GPT-4 availability
    if not hasattr(st.session_state, 'gpt4_available'):
        st.session_state.gpt4_available = None  # Unknown initially
    
    # If we know GPT-4 is available, add those models
    if st.session_state.gpt4_available is True:
        available_models = gpt4_models + base_models
    # If we don't know yet, add them but they might fail
    elif st.session_state.gpt4_available is None:
        available_models = gpt4_models[:1] + base_models  # Only add one GPT-4 to test
    
    return available_models

def initialize_openai_client():
    """Initialize OpenAI client with version compatibility"""
    try:
        if OPENAI_NEW_VERSION:
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
        else:
            # Legacy version
            openai.api_key = Config.OPENAI_API_KEY
            client = openai
        return client
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        st.info("""
        Troubleshooting steps:
        1. Check your OpenAI API key in config.py
        2. Update OpenAI package: pip install --upgrade openai
        3. Restart the application
        """)
        return None

def get_system_prompt():
    """Get enhanced system prompt for AI assistant"""
    
    base_prompt = """You are an expert data analyst AI assistant with advanced capabilities. Your role is to help users with:

CORE CAPABILITIES:
1. Advanced data analysis and statistical insights
2. SQL query generation, optimization, and explanation
3. Python code for data manipulation, visualization, and machine learning
4. Statistical analysis and hypothesis testing
5. Business intelligence and strategic recommendations
6. Data cleaning, preprocessing, and feature engineering

RESPONSE GUIDELINES:
- Always provide practical, executable code when relevant
- Include detailed explanations with code
- Suggest multiple approaches when applicable
- Consider performance and scalability
- Provide visualizations when helpful
- Include error handling in code
- Suggest next steps and further analysis

CODE STANDARDS:
- Use pandas, numpy, matplotlib, seaborn, plotly for Python
- Include proper error handling
- Add comments for complex logic
- Use efficient data processing techniques
- Include data validation steps
"""
    
    # Add enhanced data context if available
    if st.session_state.get('processed_data') and st.session_state.get('current_file'):
        selected_file = st.session_state.current_file
        data_info = st.session_state.processed_data[selected_file]
        df = data_info['dataframe']
        
        # Enhanced data profiling
        data_context = f"""

CURRENT DATA CONTEXT:
- File: {selected_file}
- Shape: {df.shape[0]:,} rows, {df.shape[1]:,} columns
- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

COLUMN ANALYSIS:
{get_detailed_column_analysis(df)}

DATA QUALITY:
- Missing Values: {df.isnull().sum().sum()} total
- Duplicate Rows: {df.duplicated().sum()}
- Data Types: {dict(df.dtypes.value_counts())}

When generating analysis or code, always consider this specific data context.
Provide insights and code that are directly applicable to this dataset.
"""
        base_prompt += data_context
    
    return base_prompt

def get_detailed_column_analysis(df):
    """Generate detailed column analysis for system prompt"""
    analysis = []
    
    for col in df.columns:
        dtype = df[col].dtype
        missing = df[col].isnull().sum()
        unique = df[col].nunique()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().any():  # Check if there are non-null values
                try:
                    stats = f"min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
                except:
                    stats = "stats unavailable"
            else:
                stats = "all null values"
        else:
            if df[col].notna().any():
                try:
                    top_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'
                    stats = f"top_value={str(top_value)[:30]}..."
                except:
                    stats = "stats unavailable"
            else:
                stats = "all null values"
        
        analysis.append(f"  - {col}: {dtype}, missing={missing}/{len(df)} ({missing/len(df)*100:.1f}%), unique={unique}, {stats}")
    
    return "\n".join(analysis)

def display_chat_messages():
    """Enhanced chat message display with code execution"""
    for i, message in enumerate(st.session_state.ai_messages):
        if message["role"] != "system":  # Don't display system message
            with st.chat_message(message["role"]):
                content = message["content"]
                
                # Enhanced content rendering with interactive elements
                render_message_content(content, i)

def render_message_content(content, message_index):
    """Render message content with enhanced features"""
    
    # Check for code blocks
    if "```" in content:
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text
                st.markdown(part)
            else:  # Code block
                # Extract language and code
                lines = part.strip().split('\n')
                if lines and lines[0] in ['sql', 'python', 'r', 'json', 'javascript']:
                    language = lines[0]
                    code = '\n'.join(lines[1:])
                else:
                    language = 'python'  # Default to python
                    code = part
                
                # Display code with copy button
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.code(code, language=language)
                with col2:
                    if st.button("📋", key=f"copy_{message_index}_{i}", help="Copy to clipboard"):
                        st.session_state.clipboard = code
                        st.success("Copied!")
                
                # Add execute button for Python code
                if language == 'python' and st.session_state.get('processed_data'):
                    if st.button("▶️ Run", key=f"run_{message_index}_{i}", help="Execute this code"):
                        execute_python_code(code, message_index, i)
    else:
        st.markdown(content)

def execute_python_code(code, message_index, code_index):
    """Execute Python code safely"""
    try:
        # Create a safe execution environment
        local_vars = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'st': st
        }
        
        # Add current data to execution context
        if st.session_state.get('processed_data') and st.session_state.get('current_file'):
            selected_file = st.session_state.current_file
            df = st.session_state.processed_data[selected_file]['dataframe']
            local_vars['df'] = df
            local_vars['data'] = df
        
        # Create a dedicated output area
        output_container = st.container()
        
        with output_container:
            st.info("🚀 Executing code...")
            
            # Redirect stdout to capture print statements
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                # Execute the code
                exec(code, globals(), local_vars)
                
                # Get the captured output
                output = captured_output.getvalue()
                
                if output:
                    st.text_area("Execution Output", output, height=200)
                
                st.success("✅ Code executed successfully!")
                
                # Check if any plots were created
                if plt.get_fignums():
                    st.subheader("📊 Generated Plots")
                    for fig_num in plt.get_fignums():
                        fig = plt.figure(fig_num)
                        st.pyplot(fig)
                
            finally:
                # Restore stdout
                sys.stdout = old_stdout
                plt.close('all')  # Close all figures to avoid memory issues
        
        # Store results
        result_key = f"execution_result_{message_index}_{code_index}"
        st.session_state[result_key] = "Code executed successfully!"
        
    except Exception as e:
        st.error(f"❌ Error executing code: {str(e)}")
        st.info("💡 Tip: Make sure the code is compatible with your current data structure")

def process_user_message(client, user_input, model, temperature, max_tokens):
    """Process user message with enhanced features"""
    
    # Add user message to chat
    st.session_state.ai_messages.append({"role": "user", "content": user_input})
    
    # Show typing indicator
    with st.spinner("🤔 Analyzing your request..."):
        try:
            response = get_ai_response(client, user_input, model, temperature, max_tokens)
            st.session_state.ai_messages.append({"role": "assistant", "content": response})
            st.rerun()
        except Exception as e:
            error_msg = str(e)
            
            # Handle model not found error specifically
            if "model_not_found" in error_msg or "gpt-4" in error_msg.lower():
                st.error("🚫 GPT-4 access not available. Switching to GPT-3.5 Turbo.")
                st.session_state.gpt4_available = False
                
                # Retry with GPT-3.5 Turbo
                fallback_model = "gpt-3.5-turbo"
                try:
                    response = get_ai_response(client, user_input, fallback_model, temperature, max_tokens)
                    st.session_state.ai_messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as retry_error:
                    st.error(f"Error with fallback model: {str(retry_error)}")
            else:
                st.error(f"Error getting AI response: {error_msg}")

def get_ai_response(client, user_input, model="gpt-3.5-turbo", temperature=0.7, max_tokens=2000):
    """Enhanced AI response with version compatibility and model fallback"""
    
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model=model,
                messages=st.session_state.ai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        else:
            # Legacy version
            response = client.ChatCompletion.create(
                model=model,
                messages=st.session_state.ai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    except Exception as e:
        # Re-raise the exception with more context
        error_msg = str(e)
        if hasattr(e, 'response') and hasattr(e.response, 'json'):
            error_data = e.response.json()
            if 'error' in error_data:
                error_msg = f"Error code: {e.response.status_code} - {error_data['error']}"
        
        raise Exception(error_msg)

def display_quick_actions_panel():
    """Enhanced quick actions panel"""
    
    st.write("### 🚀 Quick Actions")
    
    # Analysis category selection
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Data Exploration", "Machine Learning", "Visualization", "Business Insights", "Data Quality", "SQL Queries"]
    )
    
    # Dynamic actions based on selection
    if analysis_type == "Data Exploration":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 EDA Report", use_container_width=True):
                st.session_state.quick_action = "comprehensive_eda"
        with col2:
            if st.button("📈 Stats Summary", use_container_width=True):
                st.session_state.quick_action = "statistical_summary"
    
    elif analysis_type == "Machine Learning":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 ML Prep", use_container_width=True):
                st.session_state.quick_action = "ml_preparation"
        with col2:
            if st.button("⚡ Features", use_container_width=True):
                st.session_state.quick_action = "feature_engineering"
    
    elif analysis_type == "Visualization":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Advanced Viz", use_container_width=True):
                st.session_state.quick_action = "advanced_visualizations"
        with col2:
            if st.button("📈 Basic Charts", use_container_width=True):
                st.session_state.quick_action = "basic_visualizations"
    
    elif analysis_type == "Business Insights":
        if st.button("💼 Biz Analysis", use_container_width=True):
            st.session_state.quick_action = "business_insights"
    
    elif analysis_type == "Data Quality":
        if st.button("🧹 Data Cleaning", use_container_width=True):
            st.session_state.quick_action = "data_cleaning"
    
    elif analysis_type == "SQL Queries":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗃️ Generate SQL", use_container_width=True):
                st.session_state.quick_action = "sql_queries"
        with col2:
            if st.button("🔍 Analyze SQL", use_container_width=True):
                st.session_state.quick_action = "sql_analysis"

def display_data_context_panel():
    """Enhanced data context panel"""
    
    if st.session_state.get('processed_data') and st.session_state.get('current_file'):
        st.write("### 📁 Current Data Context")
        
        selected_file = st.session_state.current_file
        data_info = st.session_state.processed_data[selected_file]
        df = data_info['dataframe']
        
        # Enhanced data metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", f"{df.shape[1]:,}")
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.metric("Missing Values", df.isnull().sum().sum())
            st.metric("Duplicate Rows", df.duplicated().sum())
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Cols", numeric_cols)
        
        # Quick data preview
        with st.expander("🔍 Data Preview"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Column summary with expander
        with st.expander("📋 Column Details"):
            for col in df.columns:
                col_info = f"**{col}** (`{df[col].dtype}`) | "
                col_info += f"Missing: {df[col].isnull().sum()} | "
                col_info += f"Unique: {df[col].nunique()}"
                
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
                    col_info += f" | Range: {df[col].min():.2f} - {df[col].max():.2f}"
                
                st.write(col_info)
    else:
        st.warning("📝 No data loaded yet!")
        st.info("Please upload and process data in the Data Processing section first.")

def display_code_execution_panel():
    """Code execution and management panel"""
    
    st.write("### 💻 Code Templates")
    
    # Code templates
    template = st.selectbox(
        "Quick Templates",
        ["Select template...", "Data Cleaning", "Basic Visualization", "Statistical Test", "ML Model", "SQL Query"]
    )
    
    if template != "Select template...":
        apply_code_template(template)
    
    # Execution tips
    with st.expander("💡 Execution Tips"):
        st.markdown("""
        - **Code runs in a safe environment** with your current data as `df`
        - **Available libraries**: pandas, numpy, matplotlib, seaborn, plotly
        - **Visualizations** are automatically displayed
        - **Print statements** are captured and shown
        - **Errors** are caught and displayed with helpful messages
        """)

def apply_code_template(template_name):
    """Apply code templates"""
    
    templates = {
        "Data Cleaning": """
# Data Cleaning Template
import pandas as pd
import numpy as np

def clean_data(df):
    \"\"\"Comprehensive data cleaning function\"\"\"
    print("=== DATA CLEANING REPORT ===")
    print(f"Original shape: {df.shape}")
    
    # 1. Handle missing values
    missing_before = df.isnull().sum().sum()
    print(f"Missing values before: {missing_before}")
    
    # Strategy: Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                print(f"Filled {col} with median: {fill_value}")
            else:
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(fill_value)
                print(f"Filled {col} with mode: {fill_value}")
    
    # 2. Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate rows")
    
    # 3. Data type optimization
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Low cardinality
            df[col] = df[col].astype('category')
            print(f"Converted {col} to category")
    
    print(f"Final shape: {df.shape}")
    print(f"Missing values after: {df.isnull().sum().sum()}")
    return df

# Execute cleaning
if 'df' in locals():
    cleaned_df = clean_data(df.copy())
    print("🎉 Data cleaning completed successfully!")
else:
    print("❌ No DataFrame found. Please load your data first.")
""",
        "Basic Visualization": """
# Basic Visualization Template
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def create_basic_visualizations(df):
    \"\"\"Create comprehensive basic visualizations\"\"\"
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    print("=== BASIC VISUALIZATIONS ===")
    
    # 1. Numerical columns distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"Creating histograms for {len(numeric_cols)} numeric columns...")
        
        # Calculate subplot layout
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    else:
        print("No numeric columns found for histograms")
    
    # 2. Categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print(f"Creating bar charts for {len(categorical_cols)} categorical columns...")
        for col in categorical_cols[:3]:  # Show first 3
            value_counts = df[col].value_counts().head(10)  # Top 10
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar', color='lightcoral')
            plt.title(f'Top 10 Values in {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
    else:
        print("No categorical columns found for bar charts")

# Generate visualizations
if 'df' in locals():
    create_basic_visualizations(df)
else:
    print("❌ No DataFrame found. Please load your data first.")
"""
    }
    
    if template_name in templates:
        st.code(templates[template_name], language='python')
        if st.button(f"Apply {template_name} Template"):
            st.session_state.ai_messages.append({
                "role": "assistant", 
                "content": f"Here's a {template_name} template:\n\n```python\n{templates[template_name]}\n```"
            })
            st.rerun()

def export_chat_history():
    """Export chat history to file"""
    
    chat_text = "AI Assistant Chat History\n"
    chat_text += "=" * 50 + "\n\n"
    chat_text += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for message in st.session_state.ai_messages:
        if message["role"] != "system":
            role_icon = "👤" if message["role"] == "user" else "🤖"
            chat_text += f"{role_icon} {message['role'].upper()}:\n"
            chat_text += f"{message['content']}\n\n"
            chat_text += "-" * 50 + "\n\n"
    
    # Create download link
    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name=f"ai_assistant_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def handle_quick_actions(client, model, temperature, max_tokens):
    """Handle quick action requests"""
    if hasattr(st.session_state, 'quick_action'):
        action = st.session_state.quick_action
        action_handlers = {
            "comprehensive_eda": generate_comprehensive_eda,
            "statistical_summary": generate_statistical_summary,
            "ml_preparation": generate_ml_preparation,
            "feature_engineering": generate_feature_engineering,
            "advanced_visualizations": generate_advanced_visualizations,
            "basic_visualizations": generate_basic_visualizations,
            "business_insights": generate_business_insights,
            "data_cleaning": generate_data_cleaning,
            "sql_queries": generate_sql_queries,
            "sql_analysis": generate_sql_analysis
        }
        
        if action in action_handlers:
            # Use GPT-3.5 Turbo for quick actions if GPT-4 is not available
            effective_model = "gpt-3.5-turbo" if st.session_state.get('gpt4_available') is False else model
            action_handlers[action](client, effective_model, temperature, max_tokens)
            del st.session_state.quick_action

def generate_comprehensive_eda(client, model, temperature, max_tokens):
    """Generate comprehensive exploratory data analysis"""
    
    prompt = """
    Perform a comprehensive Exploratory Data Analysis (EDA) on the current dataset.
    Include complete Python code for:
    
    1. DATA OVERVIEW:
       - Basic statistics and info
       - Data types and memory usage
       - Missing values analysis
    
    2. UNIVARIATE ANALYSIS:
       - Distribution of numerical variables
       - Frequency of categorical variables
       - Outlier detection
    
    3. BIVARIATE ANALYSIS:
       - Correlation analysis
       - Relationship between key variables
       - Cross-tabulations
    
    4. DATA QUALITY ASSESSMENT:
       - Data quality issues
       - Cleaning recommendations
       - Feature engineering ideas
    
    Provide complete, executable Python code with plotly visualizations.
    Include detailed interpretations and business insights.
    Make the code robust with error handling.
    """
    
    process_quick_action(client, prompt, "Generate comprehensive EDA", model, temperature, max_tokens)

def generate_statistical_summary(client, model, temperature, max_tokens):
    """Generate statistical summary"""
    
    prompt = """
    Create a comprehensive statistical summary and analysis for the current dataset.
    Include:
    
    1. DESCRIPTIVE STATISTICS for all columns
    2. CORRELATION ANALYSIS with heatmap
    3. DISTRIBUTION ANALYSIS with normality tests
    4. OUTLIER DETECTION using multiple methods
    5. HYPOTHESIS TESTING examples
    
    Provide complete Python code with scipy, numpy, and seaborn.
    Include interpretations and business implications.
    """
    
    process_quick_action(client, prompt, "Generate statistical summary", model, temperature, max_tokens)

def generate_ml_preparation(client, model, temperature, max_tokens):
    """Generate machine learning preparation code"""
    
    prompt = """
    Prepare the current dataset for machine learning with comprehensive preprocessing.
    Include complete Python code for:
    
    1. FEATURE ENGINEERING and creation
    2. DATA PREPROCESSING and cleaning
    3. FEATURE SELECTION and importance
    4. DATA SPLITTING strategies
    5. BASELINE MODEL implementations
    
    Provide production-ready code with proper pipelines and error handling.
    Use scikit-learn and include evaluation metrics.
    """
    
    process_quick_action(client, prompt, "Prepare for machine learning", model, temperature, max_tokens)

def generate_feature_engineering(client, model, temperature, max_tokens):
    """Generate feature engineering code"""
    
    prompt = """
    Create comprehensive feature engineering pipeline for the current dataset.
    Include:
    
    1. AUTOMATED FEATURE GENERATION:
       - Polynomial features
       - Interaction terms
       - Time-based features
    
    2. DOMAIN-SPECIFIC FEATURES:
       - Business-specific metrics
       - Industry-standard calculations
       - Custom aggregations
    
    3. ADVANCED TECHNIQUES:
       - Feature selection algorithms
       - Dimensionality reduction
       - Automated feature importance
    
    Provide scalable, production-ready code with proper documentation.
    """
    
    process_quick_action(client, prompt, "Generate feature engineering", model, temperature, max_tokens)

def generate_advanced_visualizations(client, model, temperature, max_tokens):
    """Generate advanced visualization code"""
    
    prompt = """
    Create advanced, publication-quality visualizations for the current dataset.
    Include complete Python code for:
    
    1. INTERACTIVE VISUALIZATIONS using Plotly
    2. ADVANCED PLOTS: pair plots, correlation heatmaps, violin plots
    3. BUSINESS DASHBOARD components
    4. PROFESSIONAL STYLING and color schemes
    
    Provide complete, ready-to-run code with detailed styling.
    Ensure visualizations are informative and aesthetically pleasing.
    """
    
    process_quick_action(client, prompt, "Generate advanced visualizations", model, temperature, max_tokens)

def generate_basic_visualizations(client, model, temperature, max_tokens):
    """Generate basic visualization code"""
    
    prompt = """
    Create basic but comprehensive visualizations for the current dataset.
    Include:
    
    1. HISTOGRAMS for numerical variables
    2. BAR CHARTS for categorical variables
    3. SCATTER PLOTS for relationships
    4. BOX PLOTS for distribution analysis
    
    Use matplotlib and seaborn for static plots.
    Provide clean, well-commented code.
    """
    
    process_quick_action(client, prompt, "Generate basic visualizations", model, temperature, max_tokens)

def generate_business_insights(client, model, temperature, max_tokens):
    """Generate business insights"""
    
    prompt = """
    Provide comprehensive business intelligence analysis for the current dataset.
    Include:
    
    1. STRATEGIC ANALYSIS:
       - Key performance indicators (KPIs)
       - Trend identification
       - Opportunity analysis
    
    2. COMPETITIVE INTELLIGENCE:
       - Benchmarking analysis
       - Market positioning
       - Competitive advantages
    
    3. PREDICTIVE INSIGHTS:
       - Future trend predictions
       - Risk assessment
       - Opportunity forecasting
    
    4. ACTIONABLE RECOMMENDATIONS with specific implementation steps
    
    Provide data-driven insights with clear business implications.
    Include Python code to calculate key business metrics.
    """
    
    process_quick_action(client, prompt, "Generate business insights", model, temperature, max_tokens)

def generate_data_cleaning(client, model, temperature, max_tokens):
    """Generate data cleaning code"""
    
    prompt = """
    Create comprehensive data cleaning and preprocessing pipeline.
    Include:
    
    1. MISSING VALUE HANDLING with multiple strategies
    2. OUTLIER DETECTION AND TREATMENT
    3. DATA TYPE CONVERSION AND OPTIMIZATION
    4. DUPLICATE REMOVAL AND DATA VALIDATION
    5. DATA QUALITY METRICS AND REPORTING
    
    Provide production-ready code with comprehensive error handling.
    Include data quality reports and validation checks.
    """
    
    process_quick_action(client, prompt, "Generate data cleaning", model, temperature, max_tokens)

def generate_sql_queries(client, model, temperature, max_tokens):
    """Generate SQL queries"""
    
    prompt = """
    Generate advanced SQL queries for comprehensive data analysis of the current dataset.
    Include:
    
    1. COMPLEX ANALYTICS with window functions
    2. DATA AGGREGATION AND REPORTING QUERIES
    3. PERFORMANCE-OPTIMIZED QUERIES
    4. DATA QUALITY AND VALIDATION QUERIES
    5. BUSINESS INTELLIGENCE QUERIES
    
    Provide complete SQL code with detailed explanations.
    Include optimization tips and performance considerations.
    """
    
    process_quick_action(client, prompt, "Generate SQL queries", model, temperature, max_tokens)

def generate_sql_analysis(client, model, temperature, max_tokens):
    """Generate SQL analysis"""
    
    prompt = """
    Analyze and optimize SQL queries for the current dataset structure.
    Provide:
    
    1. QUERY PERFORMANCE ANALYSIS
    2. INDEXING RECOMMENDATIONS
    3. QUERY OPTIMIZATION TECHNIQUES
    4. BEST PRACTICES FOR THE CURRENT DATA SCHEMA
    5. ALTERNATIVE QUERY APPROACHES
    
    Focus on practical, actionable advice for improving SQL performance.
    """
    
    process_quick_action(client, prompt, "Analyze SQL queries", model, temperature, max_tokens)

def process_quick_action(client, prompt, action_name, model, temperature, max_tokens):
    """Process quick action requests"""
    
    with st.spinner(f"🔄 {action_name}..."):
        try:
            response = get_ai_response(client, prompt, model, temperature, max_tokens)
            st.session_state.ai_messages.extend([
                {"role": "user", "content": action_name},
                {"role": "assistant", "content": response}
            ])
            st.rerun()
        except Exception as e:
            error_msg = str(e)
            
            # Handle model not found error
            if "model_not_found" in error_msg or "gpt-4" in error_msg.lower():
                st.error("🚫 GPT-4 not available for this action. Switching to GPT-3.5 Turbo.")
                st.session_state.gpt4_available = False
                
                # Retry with GPT-3.5 Turbo
                fallback_model = "gpt-3.5-turbo"
                try:
                    response = get_ai_response(client, prompt, fallback_model, temperature, max_tokens)
                    st.session_state.ai_messages.extend([
                        {"role": "user", "content": action_name},
                        {"role": "assistant", "content": response}
                    ])
                    st.rerun()
                except Exception as retry_error:
                    st.error(f"Error with fallback model: {str(retry_error)}")
            else:
                st.error(f"Error in {action_name}: {error_msg}")

def extract_code_from_response(response, language='python'):
    """Extract code blocks from AI response"""
    code_blocks = re.findall(f'```{language}(.*?)```', response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(f'```(.*?)```', response, re.DOTALL)
    
    return [block.strip() for block in code_blocks]