import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import tempfile
import os
import re
from difflib import SequenceMatcher
from scipy.stats import pearsonr

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    st.warning("Pyvis not available. Interactive network diagrams will be limited.")

def render_multi_file_analysis():
    """Render enhanced multi-file relationship analysis"""
    
    st.header("🔗 Enhanced Multi-File Relationship Analysis")
    
    # Display guidelines
    display_guidelines()
    
    if len(st.session_state.processed_data) < 2:
        st.warning("📁 Please upload at least 2 files for multi-file analysis")
        return
    
    # File overview section
    display_file_overview()
    
    st.markdown("---")
    
    # Analysis controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        similarity_threshold = st.slider(
            "Column Similarity Threshold", 
            min_value=0.5, 
            max_value=1.0, 
            value=0.8,
            help="Threshold for fuzzy column name matching"
        )
    
    with col2:
        min_overlap_threshold = st.slider(
            "Minimum Overlap %", 
            min_value=0, 
            max_value=100, 
            value=10,
            help="Minimum overlap percentage to consider relationships"
        )
    
    with col3:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["Basic", "Comprehensive", "Deep Analysis"],
            help="Basic: Column names only, Comprehensive: +Fuzzy matching, Deep: +Statistical analysis"
        )
    
    if st.button("🚀 Analyze Schemas & Relationships", type="primary"):
        with st.spinner("🔍 Analyzing file schemas, relationships, and generating insights..."):
            try:
                # Perform comprehensive analysis
                schema_analysis = analyze_schemas()
                relationship_analysis = analyze_relationships(
                    similarity_threshold, 
                    min_overlap_threshold,
                    analysis_depth
                )
                statistical_analysis = analyze_statistical_relationships() if analysis_depth == "Deep Analysis" else {}
                data_quality_report = generate_data_quality_report()
                
                # Display results
                display_schema_analysis(schema_analysis)
                display_relationship_analysis(relationship_analysis, statistical_analysis)
                display_data_quality_report(data_quality_report)
                
                # Generate diagrams
                st.subheader("🕸️ Relationship & Schema Diagrams")
                generate_relationship_diagram(relationship_analysis)
                generate_schema_diagram(schema_analysis, relationship_analysis)
                
                # Integration recommendations
                display_integration_recommendations(relationship_analysis, schema_analysis)
                
                # Export functionality
                display_export_options(relationship_analysis, schema_analysis, data_quality_report)
                
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.info("💡 Try reducing the number of files or adjusting analysis parameters")

def display_guidelines():
    """Display expandable guidelines for users"""
    with st.expander("📚 **Guidelines - How to Use Multi-File Analysis**", expanded=False):
        st.markdown("""
        ### 🎯 Purpose
        This tool helps you understand relationships between multiple datasets and provides insights for data integration.
        
        ### 🔍 What You'll Get
        
        **1. Schema Analysis**
        - Column data types and statistics
        - Primary key candidates
        - Data quality metrics
        
        **2. Relationship Detection**
        - **One-to-One (1:1)**: Each record in File A matches exactly one record in File B
        - **One-to-Many (1:N)**: One record in File A matches multiple records in File B
        - **Many-to-Many (M:N)**: Multiple records in File A match multiple records in File B
        - **Partial Overlap**: Some common values with no clear relationship pattern
        
        **3. Advanced Features**
        - Fuzzy column name matching
        - Statistical correlation analysis
        - Data quality assessment
        - Integration recommendations
        
        ### ⚙️ Configuration Tips
        
        **Similarity Threshold**: Higher values require more exact column name matches
        **Minimum Overlap**: Filter out weak relationships
        **Analysis Depth**: Choose based on your needs:
        - **Basic**: Quick analysis using exact column names
        - **Comprehensive**: Includes fuzzy matching and basic statistics
        - **Deep Analysis**: Full statistical correlation analysis
        
        ### 💡 Best Practices
        1. Upload clean, well-structured data files
        2. Start with Basic analysis, then move to Comprehensive
        3. Review primary key candidates for data modeling
        4. Use integration recommendations for JOIN operations
        """)

def display_file_overview():
    """Display enhanced file information overview"""
    st.write("### 📁 Uploaded Files Overview")
    
    files_info = []
    for file_name, file_data in st.session_state.processed_data.items():
        df = file_data['dataframe']
        files_info.append({
            'File Name': file_name,
            'Rows': len(df),
            'Columns': len(df.columns),
            'Size (MB)': round(file_data['file_size'] / (1024 * 1024), 2),
            'Missing Values': df.isnull().sum().sum(),
            'Duplicate Rows': df.duplicated().sum()
        })
    
    files_df = pd.DataFrame(files_info)
    st.dataframe(files_df, use_container_width=True)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", len(files_info))
    with col2:
        st.metric("Total Columns", sum(info['Columns'] for info in files_info))
    with col3:
        st.metric("Total Rows", sum(info['Rows'] for info in files_info))
    with col4:
        total_size = sum(info['Size (MB)'] for info in files_info)
        st.metric("Total Size", f"{total_size:.2f} MB")

def analyze_schemas():
    """Enhanced schema analysis with data profiling"""
    schema_info = {}
    
    for file_name, file_data in st.session_state.processed_data.items():
        df = file_data['dataframe']
        
        # Basic schema info with robust data type handling
        schema_info[file_name] = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'primary_key_candidates': find_primary_key_candidates(df),
            'numeric_columns': get_numeric_columns(df),
            'categorical_columns': get_categorical_columns(df),
            'date_columns': get_date_columns(df),
            'data_profile': generate_data_profile(df),
            'column_stats': generate_column_statistics(df)
        }
    
    return schema_info

def get_numeric_columns(df):
    """Safely get numeric columns"""
    try:
        return list(df.select_dtypes(include=[np.number]).columns)
    except:
        # Fallback: manually check for numeric types
        numeric_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
            except:
                continue
        return numeric_cols

def get_categorical_columns(df):
    """Safely get categorical columns"""
    try:
        # Use object dtype and exclude numeric/datetime
        cat_cols = []
        for col in df.columns:
            try:
                if (pd.api.types.is_object_dtype(df[col]) or 
                    pd.api.types.is_string_dtype(df[col])):
                    cat_cols.append(col)
            except:
                continue
        return cat_cols
    except:
        return []

def get_date_columns(df):
    """Safely get date columns"""
    try:
        return list(df.select_dtypes(include=['datetime', 'datetimetz']).columns)
    except:
        # Fallback: manually check for datetime types
        date_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)
            except:
                continue
        return date_cols

def find_primary_key_candidates(df):
    """Enhanced primary key detection with multiple criteria"""
    candidates = []
    
    for col in df.columns:
        try:
            uniqueness = df[col].nunique() / len(df) if len(df) > 0 else 0
            completeness = df[col].notna().mean() if len(df) > 0 else 0
            
            # Enhanced scoring
            score = (uniqueness * 0.6 + completeness * 0.4)
            
            if uniqueness == 1.0 and completeness == 1.0:
                candidates.append({
                    'column': col,
                    'uniqueness': uniqueness,
                    'completeness': completeness,
                    'score': 1.0,
                    'type': 'Perfect Primary Key',
                    'data_type': str(df[col].dtype)
                })
            elif score > 0.95:
                candidates.append({
                    'column': col,
                    'uniqueness': uniqueness,
                    'completeness': completeness,
                    'score': score,
                    'type': 'Excellent Candidate',
                    'data_type': str(df[col].dtype)
                })
            elif score > 0.85:
                candidates.append({
                    'column': col,
                    'uniqueness': uniqueness,
                    'completeness': completeness,
                    'score': score,
                    'type': 'Good Candidate',
                    'data_type': str(df[col].dtype)
                })
        except:
            continue
    
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[:5]  # Top 5 candidates

def generate_data_profile(df):
    """Generate comprehensive data profile"""
    try:
        total_rows = len(df)
        total_columns = len(df.columns)
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_columns)) * 100 if (total_rows * total_columns) > 0 else 0
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        profile = {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': duplicate_percentage,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        }
    except:
        profile = {
            'total_rows': 0,
            'total_columns': 0,
            'missing_values': 0,
            'missing_percentage': 0,
            'duplicate_rows': 0,
            'duplicate_percentage': 0,
            'memory_usage': 0
        }
    
    return profile

def generate_column_statistics(df):
    """Generate statistics for each column with robust error handling"""
    stats = {}
    for col in df.columns:
        try:
            col_stats = {
                'data_type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100 if len(df) > 0 else 0,
                'unique_count': df[col].nunique(),
                'unique_percentage': (df[col].nunique() / len(df)) * 100 if len(df) > 0 else 0
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    col_stats.update({
                        'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                        'std': float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                        'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                        'max': float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                        'median': float(df[col].median()) if not pd.isna(df[col].median()) else 0
                    })
                except (TypeError, ValueError):
                    pass
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    col_stats.update({
                        'min_date': df[col].min(),
                        'max_date': df[col].max()
                    })
                except:
                    pass
            else:
                # Categorical columns
                try:
                    top_values = df[col].value_counts().head(3)
                    col_stats['top_values'] = top_values.to_dict()
                except:
                    col_stats['top_values'] = {}
            
            stats[col] = col_stats
        except:
            # Skip columns that cause errors
            continue
    
    return stats

def analyze_relationships(similarity_threshold=0.8, min_overlap_threshold=10, analysis_depth="Basic"):
    """Enhanced relationship analysis with fuzzy matching"""
    relationships = []
    files = list(st.session_state.processed_data.keys())
    
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i < j:  # Avoid duplicate comparisons
                try:
                    df1 = st.session_state.processed_data[file1]['dataframe']
                    df2 = st.session_state.processed_data[file2]['dataframe']
                    
                    # Find matching columns with fuzzy matching
                    matching_pairs = find_matching_columns(df1, df2, similarity_threshold, analysis_depth)
                    
                    for col1, col2, similarity in matching_pairs:
                        try:
                            relationship = analyze_column_relationship(df1, df2, col1, col2, file1, file2, min_overlap_threshold)
                            if relationship:
                                relationship['similarity_score'] = similarity
                                relationships.append(relationship)
                        except:
                            continue
                except:
                    continue
    
    return relationships

def find_matching_columns(df1, df2, similarity_threshold, analysis_depth):
    """Find matching columns using exact and fuzzy matching"""
    matches = []
    try:
        cols1 = df1.columns.tolist()
        cols2 = df2.columns.tolist()
        
        # Exact matches
        exact_matches = set(cols1) & set(cols2)
        for col in exact_matches:
            matches.append((col, col, 1.0))
        
        if analysis_depth in ["Comprehensive", "Deep Analysis"]:
            # Fuzzy matches
            for col1 in cols1:
                for col2 in cols2:
                    if col1 != col2 and (col1, col2) not in matches and (col2, col1) not in matches:
                        similarity = calculate_similarity(col1, col2)
                        if similarity >= similarity_threshold:
                            matches.append((col1, col2, similarity))
    except:
        pass
    
    return matches

def calculate_similarity(a, b):
    """Calculate similarity between two strings"""
    try:
        a_clean = re.sub(r'[^a-zA-Z0-9]', '', str(a).lower())
        b_clean = re.sub(r'[^a-zA-Z0-9]', '', str(b).lower())
        return SequenceMatcher(None, a_clean, b_clean).ratio()
    except:
        return 0.0

def analyze_column_relationship(df1, df2, col1, col2, file1, file2, min_overlap_threshold):
    """Enhanced relationship analysis with improved typing"""
    
    # Check data type compatibility with more flexibility
    if not is_data_type_compatible(df1[col1], df2[col2]):
        return None
    
    try:
        # Remove nulls for analysis and convert to string for comparison
        vals1 = df1[col1].dropna().astype(str)
        vals2 = df2[col2].dropna().astype(str)
        
        if len(vals1) == 0 or len(vals2) == 0:
            return None
        
        # Calculate overlap
        unique1 = set(vals1.unique())
        unique2 = set(vals2.unique())
        overlap = unique1 & unique2
        
        if len(overlap) == 0:
            return None
        
        overlap_pct1 = len(overlap) / len(unique1) * 100 if len(unique1) > 0 else 0
        overlap_pct2 = len(overlap) / len(unique2) * 100 if len(unique2) > 0 else 0
        
        if max(overlap_pct1, overlap_pct2) < min_overlap_threshold:
            return None
        
        # Enhanced relationship type determination
        relationship_type = determine_enhanced_relationship_type(vals1, vals2, overlap)
        
        return {
            'file1': file1,
            'file2': file2,
            'column1': col1,
            'column2': col2,
            'relationship_type': relationship_type,
            'overlap_count': len(overlap),
            'overlap_percentage_file1': overlap_pct1,
            'overlap_percentage_file2': overlap_pct2,
            'unique1_count': len(unique1),
            'unique2_count': len(unique2),
            'data_type1': str(df1[col1].dtype),
            'data_type2': str(df2[col2].dtype)
        }
    except:
        return None

def is_data_type_compatible(series1, series2):
    """Check if two series have compatible data types with robust error handling"""
    try:
        dtype1 = str(series1.dtype).lower()
        dtype2 = str(series2.dtype).lower()
        
        # Basic type compatibility groups
        numeric_keywords = ['int', 'float', 'number']
        string_keywords = ['object', 'string', 'str']
        datetime_keywords = ['datetime', 'date', 'time']
        
        def check_dtype_group(dtype_str, keywords):
            return any(keyword in dtype_str for keyword in keywords)
        
        # Check if both are numeric
        if (check_dtype_group(dtype1, numeric_keywords) and 
            check_dtype_group(dtype2, numeric_keywords)):
            return True
        
        # Check if both are string-like
        if (check_dtype_group(dtype1, string_keywords) and 
            check_dtype_group(dtype2, string_keywords)):
            return True
        
        # Check if both are datetime
        if (check_dtype_group(dtype1, datetime_keywords) and 
            check_dtype_group(dtype2, datetime_keywords)):
            return True
        
        # Allow comparison between numeric and string if we can safely convert
        if ((check_dtype_group(dtype1, numeric_keywords) and check_dtype_group(dtype2, string_keywords)) or
            (check_dtype_group(dtype1, string_keywords) and check_dtype_group(dtype2, numeric_keywords))):
            return True
            
        # Final fallback - allow comparison with conversion to string
        return True
        
    except:
        # If we can't determine compatibility, allow the comparison
        return True

def determine_enhanced_relationship_type(vals1, vals2, overlap):
    """Enhanced relationship type determination"""
    try:
        unique1 = set(vals1.unique())
        unique2 = set(vals2.unique())
        
        # Calculate cardinalities
        file1_to_file2 = {}
        file2_to_file1 = {}
        
        for val in overlap:
            file1_to_file2[val] = len(vals1[vals1 == val])
            file2_to_file1[val] = len(vals2[vals2 == val])
        
        avg_file1_to_file2 = np.mean(list(file1_to_file2.values())) if file1_to_file2 else 0
        avg_file2_to_file1 = np.mean(list(file2_to_file1.values())) if file2_to_file1 else 0
        
        # Determine relationship type
        if (len(overlap) == len(unique1) == len(unique2) and 
            avg_file1_to_file2 == 1.0 and avg_file2_to_file1 == 1.0):
            return "One-to-One (1:1)"
        elif (len(overlap) == len(unique1) and 
              avg_file1_to_file2 == 1.0 and avg_file2_to_file1 >= 1.0):
            return "One-to-Many (1:N) - File1 to File2"
        elif (len(overlap) == len(unique2) and 
              avg_file2_to_file1 == 1.0 and avg_file1_to_file2 >= 1.0):
            return "One-to-Many (1:N) - File2 to File1"
        elif (avg_file1_to_file2 > 1.0 and avg_file2_to_file1 > 1.0):
            return "Many-to-Many (M:N)"
        else:
            return "Partial Overlap"
    except:
        return "Partial Overlap"

def analyze_statistical_relationships():
    """Analyze statistical relationships between numeric columns"""
    statistical_relationships = {}
    files = list(st.session_state.processed_data.keys())
    
    for i, file1 in enumerate(files):
        for j, file2 in enumerate(files):
            if i < j:
                try:
                    df1 = st.session_state.processed_data[file1]['dataframe']
                    df2 = st.session_state.processed_data[file2]['dataframe']
                    
                    # Use robust numeric detection
                    numeric_cols1 = get_numeric_columns(df1)
                    numeric_cols2 = get_numeric_columns(df2)
                    
                    correlations = []
                    for col1 in numeric_cols1:
                        for col2 in numeric_cols2:
                            # Align data by index for correlation
                            aligned_data = align_data_for_correlation(df1[col1], df2[col2])
                            if aligned_data is not None and len(aligned_data) >= 10:
                                try:
                                    corr, p_value = pearsonr(aligned_data[col1], aligned_data[col2])
                                    if not np.isnan(corr) and abs(corr) > 0.5 and p_value < 0.05:
                                        correlations.append({
                                            'column1': col1,
                                            'column2': col2,
                                            'correlation': corr,
                                            'p_value': p_value,
                                            'strength': 'Strong' if abs(corr) > 0.7 else 'Moderate'
                                        })
                                except:
                                    continue
                    
                    if correlations:
                        key = f"{file1}_{file2}"
                        statistical_relationships[key] = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
                except:
                    continue
    
    return statistical_relationships

def align_data_for_correlation(series1, series2):
    """Align two series for correlation analysis"""
    try:
        # Convert to numeric, coercing errors to NaN
        s1_numeric = pd.to_numeric(series1, errors='coerce')
        s2_numeric = pd.to_numeric(series2, errors='coerce')
        
        aligned_df = pd.DataFrame({'s1': s1_numeric, 's2': s2_numeric}).dropna()
        if len(aligned_df) < 10:  # Minimum sample size
            return None
        return aligned_df.rename(columns={'s1': series1.name, 's2': series2.name})
    except:
        return None

def generate_data_quality_report():
    """Generate comprehensive data quality report"""
    quality_report = {}
    
    for file_name, file_data in st.session_state.processed_data.items():
        try:
            df = file_data['dataframe']
            report = {
                'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 0,
                'uniqueness': (df.nunique() / len(df)).mean() * 100 if len(df) > 0 else 0,
                'consistency': calculate_consistency_score(df),
                'accuracy_indicators': find_accuracy_indicators(df)
            }
            report['overall_score'] = np.mean([report['completeness'], report['uniqueness'], report['consistency']])
            quality_report[file_name] = report
        except:
            # Skip files that cause errors
            continue
    
    return quality_report

def calculate_consistency_score(df):
    """Calculate data consistency score"""
    score = 0
    total_checks = 0
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for outliers (values beyond 3 standard deviations)
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                try:
                    z_scores = np.abs((non_null_data - non_null_data.mean()) / non_null_data.std())
                    outlier_ratio = (z_scores > 3).sum() / len(df)
                    score += (1 - outlier_ratio)
                    total_checks += 1
                except:
                    # Skip columns where std calculation fails
                    continue
    
    return (score / total_checks * 100) if total_checks > 0 else 100

def find_accuracy_indicators(df):
    """Find potential data accuracy issues"""
    issues = []
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check for negative values in typically positive fields
            if 'age' in col.lower() and (df[col] < 0).any():
                issues.append(f"Negative values in {col}")
            elif 'price' in col.lower() and (df[col] < 0).any():
                issues.append(f"Negative values in {col}")
    
    return issues if issues else ["No major accuracy issues detected"]

def display_schema_analysis(schema_analysis):
    """Enhanced schema analysis display"""
    st.write("#### 📋 Enhanced Schema Analysis")
    
    for file_name, schema in schema_analysis.items():
        with st.expander(f"📊 **{file_name}** - Detailed Schema Analysis", expanded=False):
            
            # Data Profile
            st.write("**📈 Data Profile**")
            profile = schema['data_profile']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", profile['total_rows'])
            with col2:
                st.metric("Total Columns", profile['total_columns'])
            with col3:
                st.metric("Missing Values", f"{profile['missing_percentage']:.1f}%")
            with col4:
                st.metric("Duplicate Rows", f"{profile['duplicate_percentage']:.1f}%")
            
            # Columns and Data Types
            st.write("**🔧 Columns & Data Types**")
            dtype_df = pd.DataFrame({
                'Column': list(schema['dtypes'].keys()),
                'Data Type': list(schema['dtypes'].values()),
                'Missing %': [schema['column_stats'][col]['missing_percentage'] for col in schema['dtypes'].keys()],
                'Unique %': [schema['column_stats'][col]['unique_percentage'] for col in schema['dtypes'].keys()]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
            # Primary Key Candidates
            st.write("**🔑 Primary Key Candidates**")
            if schema['primary_key_candidates']:
                pk_df = pd.DataFrame(schema['primary_key_candidates'])
                st.dataframe(pk_df, use_container_width=True)
            else:
                st.info("No strong primary key candidates found")
            
            # Column Type Summary
            st.write("**📊 Column Type Summary**")
            type_summary = {
                'Numeric': len(schema['numeric_columns']),
                'Categorical': len(schema['categorical_columns']),
                'Date': len(schema['date_columns'])
            }
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Numeric Columns", type_summary['Numeric'])
            with col2:
                st.metric("Categorical Columns", type_summary['Categorical'])
            with col3:
                st.metric("Date Columns", type_summary['Date'])

def display_relationship_analysis(relationships, statistical_relationships):
    """Enhanced relationship analysis display"""
    st.write("#### 🔗 Enhanced Relationship Analysis")
    
    if not relationships:
        st.info("🤷 No strong relationships found between files. Try adjusting similarity threshold or minimum overlap.")
        return
    
    # Create enhanced relationships dataframe
    rel_data = []
    for rel in relationships:
        rel_data.append({
            'File 1': rel['file1'],
            'File 2': rel['file2'],
            'Column 1': rel['column1'],
            'Column 2': rel['column2'],
            'Relationship Type': rel['relationship_type'],
            'Overlap % File1': f"{rel['overlap_percentage_file1']:.1f}%",
            'Overlap % File2': f"{rel['overlap_percentage_file2']:.1f}%",
            'Similarity Score': f"{rel.get('similarity_score', 1.0):.2f}",
            'Data Type 1': rel['data_type1'],
            'Data Type 2': rel['data_type2']
        })
    
    rel_df = pd.DataFrame(rel_data)
    st.dataframe(rel_df, use_container_width=True)
    
    # Summary statistics
    st.write("#### 📊 Relationship Summary")
    
    rel_type_counts = pd.Series([r['relationship_type'] for r in relationships]).value_counts()
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Relationships", len(relationships))
    with col2:
        st.metric("Relationship Types", len(rel_type_counts))
    with col3:
        avg_overlap = np.mean([r['overlap_percentage_file1'] for r in relationships]) if relationships else 0
        st.metric("Average Overlap", f"{avg_overlap:.1f}%")
    with col4:
        strong_rels = len([r for r in relationships if r['overlap_percentage_file1'] > 70])
        st.metric("Strong Relationships", strong_rels)
    
    # Relationship type distribution
    if len(rel_type_counts) > 0:
        fig_rel_type = go.Figure(data=[go.Pie(
            labels=rel_type_counts.index,
            values=rel_type_counts.values,
            hole=.3,
            textinfo='label+percent'
        )])
        fig_rel_type.update_layout(
            title="Relationship Type Distribution",
            showlegend=True
        )
        st.plotly_chart(fig_rel_type, use_container_width=True)
    
    # Statistical relationships if available
    if statistical_relationships:
        display_statistical_relationships(statistical_relationships)

def display_statistical_relationships(statistical_relationships):
    """Display statistical correlation analysis"""
    st.write("#### 📈 Statistical Relationships (Numeric Columns)")
    
    for key, correlations in statistical_relationships.items():
        file1, file2 = key.split('_')
        with st.expander(f"📊 {file1} ↔ {file2} - Statistical Correlations"):
            if correlations:
                corr_df = pd.DataFrame(correlations)
                st.dataframe(corr_df, use_container_width=True)
                
                # Display top correlation
                top_corr = correlations[0]
                st.write(f"**Strongest Correlation:** {top_corr['column1']} ↔ {top_corr['column2']}")
                st.write(f"Correlation: {top_corr['correlation']:.3f} ({top_corr['strength']})")
            else:
                st.info("No significant statistical correlations found")

def display_data_quality_report(quality_report):
    """Display data quality assessment"""
    st.write("#### 🎯 Data Quality Assessment")
    
    if not quality_report:
        st.info("No data quality information available")
        return
    
    quality_data = []
    for file_name, report in quality_report.items():
        quality_data.append({
            'File Name': file_name,
            'Completeness %': f"{report['completeness']:.1f}%",
            'Uniqueness %': f"{report['uniqueness']:.1f}%",
            'Consistency %': f"{report['consistency']:.1f}%",
            'Overall Score': f"{report['overall_score']:.1f}%",
            'Accuracy Issues': len(report['accuracy_indicators'])
        })
    
    quality_df = pd.DataFrame(quality_data)
    st.dataframe(quality_df, use_container_width=True)
    
    # Quality score visualization
    fig_quality = go.Figure()
    for file_name, report in quality_report.items():
        fig_quality.add_trace(go.Bar(
            name=file_name,
            x=['Completeness', 'Uniqueness', 'Consistency', 'Overall'],
            y=[report['completeness'], report['uniqueness'], report['consistency'], report['overall_score']]
        ))
    
    fig_quality.update_layout(
        title="Data Quality Scores by File",
        barmode='group',
        yaxis_title="Score (%)",
        yaxis_range=[0, 100]
    )
    st.plotly_chart(fig_quality, use_container_width=True)

def generate_relationship_diagram(relationships):
    """Generate enhanced interactive relationship diagram"""
    
    if not PYVIS_AVAILABLE:
        st.warning("""
        **Pyvis not installed for interactive network diagrams.**
        
        To enable interactive network visualization, install pyvis:
        ```bash
        pip install pyvis
        ```
        
        For now, here's a static visualization of the relationships:
        """)
        generate_static_relationship_diagram(relationships)
        return
    
    try:
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (files) with enhanced properties
        for file_name in st.session_state.processed_data.keys():
            file_data = st.session_state.processed_data[file_name]
            node_size = min(30 + len(file_data['dataframe'].columns) * 2, 60)
            G.add_node(file_name, 
                      size=node_size, 
                      title=f"{file_name}\nColumns: {len(file_data['dataframe'].columns)}\nRows: {len(file_data['dataframe'])}",
                      color='lightblue',
                      shape='dot')
        
        # Add edges (relationships) with enhanced properties
        for rel in relationships:
            weight = min(rel['overlap_percentage_file1'], rel['overlap_percentage_file2'])
            edge_width = max(weight / 15, 1)
            
            # Color based on relationship strength
            if weight > 80:
                edge_color = '#ff4444'  # Red for strong
            elif weight > 50:
                edge_color = '#ffaa00'  # Orange for medium
            else:
                edge_color = '#888888'  # Gray for weak
            
            title = (f"Columns: {rel['column1']} ↔ {rel['column2']}\n"
                    f"Type: {rel['relationship_type']}\n"
                    f"Overlap: {weight:.1f}%\n"
                    f"Similarity: {rel.get('similarity_score', 1.0):.2f}")
            
            G.add_edge(
                rel['file1'], 
                rel['file2'],
                weight=weight,
                title=title,
                color=edge_color,
                width=edge_width,
                smooth=True
            )
        
        # Create pyvis network
        net = Network(height='600px', width='100%', bgcolor='#ffffff', font_color='black')
        
        # Enhanced options
        net.set_options("""
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100},
            "barnesHut": {
              "gravitationalConstant": -8000,
              "springLength": 95,
              "springConstant": 0.04
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "keyboard": {"enabled": true}
          },
          "layout": {
            "improvedLayout": true
          }
        }
        """)
        
        # Add nodes and edges
        for node in G.nodes(data=True):
            net.add_node(node[0], **node[1])
        
        for edge in G.edges(data=True):
            net.add_edge(edge[0], edge[1], **edge[2])
        
        # Save and display
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            html_path = tmp.name
        
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.components.v1.html(html_content, height=600)
        
        # Clean up
        os.unlink(html_path)
        
    except Exception as e:
        st.error(f"❌ Error generating interactive diagram: {str(e)}")
        st.info("🔄 Falling back to static visualization...")
        generate_static_relationship_diagram(relationships)

def generate_schema_diagram(schema_analysis, relationships):
    """Generate schema diagram showing tables and columns"""
    st.write("#### 🗃️ Schema Diagram")
    
    try:
        # Create a new graph for schema visualization
        G = nx.Graph()
        pos = {}
        
        # Add tables and their columns
        for file_idx, (file_name, schema) in enumerate(schema_analysis.items()):
            # Add table node
            table_node = f"table_{file_idx}"
            G.add_node(table_node, 
                      label=file_name, 
                      shape='box',
                      color='lightblue',
                      level=0)
            
            pos[table_node] = (file_idx * 2, 0)
            
            # Add column nodes
            for col_idx, column in enumerate(schema['columns']):
                col_node = f"col_{file_idx}_{col_idx}"
                G.add_node(col_node, 
                          label=column, 
                          shape='ellipse',
                          color='lightgreen',
                          level=1)
                
                # FIXED: Properly closed parentheses
                x_pos = file_idx * 2 + (col_idx - len(schema['columns']) / 2) * 0.3
                y_pos = -1 - col_idx * 0.3
                pos[col_node] = (x_pos, y_pos)
                
                # Connect column to table
                G.add_edge(table_node, col_node, color='gray', style='dashed')
        
        # Add relationship edges
        for rel in relationships:
            # Find the column nodes for this relationship
            col1_node = None
            col2_node = None
            
            for file_idx, (file_name, schema) in enumerate(schema_analysis.items()):
                if file_name == rel['file1']:
                    for col_idx, column in enumerate(schema['columns']):
                        if column == rel['column1']:
                            col1_node = f"col_{file_idx}_{col_idx}"
                            break
                if file_name == rel['file2']:
                    for col_idx, column in enumerate(schema['columns']):
                        if column == rel['column2']:
                            col2_node = f"col_{file_idx}_{col_idx}"
                            break
            
            if col1_node and col2_node:
                weight = min(rel['overlap_percentage_file1'], rel['overlap_percentage_file2'])
                edge_color = '#ff4444' if weight > 80 else '#ffaa00' if weight > 50 else '#888888'
                
                G.add_edge(col1_node, col2_node, 
                          color=edge_color,
                          width=2,
                          label=f"{rel['relationship_type']}\n{weight:.1f}%")
        
        # Create Plotly visualization
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in G.edges(data=True):
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_text.append(edge[2].get('label', ''))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=edge_text * 3
        )
        
        # Node traces
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(G.nodes[node].get('label', node))
                node_color.append(G.nodes[node].get('color', 'gray'))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=20,
                color=node_color,
                line=dict(width=2, color='darkblue')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Database Schema Diagram',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500)
                       )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating schema diagram: {str(e)}")
        st.info("Schema diagram is not available for this data configuration")

def display_integration_recommendations(relationships, schema_analysis):
    """Display enhanced integration recommendations"""
    st.write("#### 💡 Enhanced Integration Recommendations")
    
    if not relationships:
        st.info("No relationships found for integration recommendations")
        return
    
    # Group relationships by file pairs
    file_pairs = {}
    for rel in relationships:
        key = (rel['file1'], rel['file2'])
        if key not in file_pairs:
            file_pairs[key] = []
        file_pairs[key].append(rel)
    
    # Display recommendations for each file pair
    for (file1, file2), rels in file_pairs.items():
        with st.expander(f"🔗 **{file1}** ↔ **{file2}** - Integration Strategy"):
            
            # Find the strongest relationship
            strongest_rel = max(rels, key=lambda x: x['overlap_percentage_file1'])
            
            st.write(f"**Primary Join Recommendation:**")
            st.code(f"SELECT *\nFROM {file1}\nJOIN {file2} \nON {file1}.{strongest_rel['column1']} = {file2}.{strongest_rel['column2']}", language='sql')
            
            st.write(f"**Relationship Details:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Relationship Type", strongest_rel['relationship_type'])
            with col2:
                st.metric("Overlap", f"{strongest_rel['overlap_percentage_file1']:.1f}%")
            with col3:
                st.metric("Data Compatibility", "✓" if strongest_rel['data_type1'] == strongest_rel['data_type2'] else "⚠")
            
            # Additional relationships
            if len(rels) > 1:
                st.write("**Alternative Join Keys:**")
                for rel in rels[:3]:  # Show top 3 alternatives
                    if rel != strongest_rel:
                        st.write(f"- `{rel['column1']}` ↔ `{rel['column2']}` ({rel['relationship_type']}, {rel['overlap_percentage_file1']:.1f}% overlap)")
            
            # Data modeling suggestions
            st.write("**Data Modeling Suggestions:**")
            if "One-to-One" in strongest_rel['relationship_type']:
                st.success("✅ Consider merging these tables or creating a unified view")
            elif "One-to-Many" in strongest_rel['relationship_type']:
                st.info("🔗 Standard parent-child relationship. Maintain referential integrity.")
            elif "Many-to-Many" in strongest_rel['relationship_type']:
                st.warning("🔄 Consider creating a junction table for this relationship")

def display_export_options(relationships, schema_analysis, data_quality_report):
    """Display options to export analysis results"""
    st.write("#### 💾 Export Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Relationship Report"):
            export_relationship_report(relationships)
    
    with col2:
        if st.button("🔄 Export SQL Joins"):
            export_sql_joins(relationships)
    
    with col3:
        if st.button("📊 Export Data Quality Report"):
            export_data_quality_report(data_quality_report)

def export_relationship_report(relationships):
    """Export relationship analysis as CSV"""
    if relationships:
        export_df = pd.DataFrame(relationships)
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Relationship Report CSV",
            data=csv,
            file_name="relationship_analysis.csv",
            mime="text/csv"
        )

def export_sql_joins(relationships):
    """Export recommended SQL joins"""
    if relationships:
        sql_queries = []
        file_pairs = {}
        
        for rel in relationships:
            key = (rel['file1'], rel['file2'])
            if key not in file_pairs:
                file_pairs[key] = []
            file_pairs[key].append(rel)
        
        for (file1, file2), rels in file_pairs.items():
            strongest_rel = max(rels, key=lambda x: x['overlap_percentage_file1'])
            sql = f"-- Join between {file1} and {file2}\n"
            sql += f"SELECT *\nFROM {file1}\nJOIN {file2} \nON {file1}.{strongest_rel['column1']} = {file2}.{strongest_rel['column2']};\n\n"
            sql_queries.append(sql)
        
        sql_content = "".join(sql_queries)
        st.download_button(
            label="📥 Download SQL Joins",
            data=sql_content,
            file_name="recommended_joins.sql",
            mime="text/plain"
        )

def export_data_quality_report(data_quality_report):
    """Export data quality report"""
    if data_quality_report:
        quality_data = []
        for file_name, report in data_quality_report.items():
            quality_data.append({
                'File Name': file_name,
                'Completeness_%': report['completeness'],
                'Uniqueness_%': report['uniqueness'],
                'Consistency_%': report['consistency'],
                'Overall_Score_%': report['overall_score'],
                'Accuracy_Issues': ', '.join(report['accuracy_indicators'])
            })
        
        quality_df = pd.DataFrame(quality_data)
        csv = quality_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data Quality Report",
            data=csv,
            file_name="data_quality_report.csv",
            mime="text/csv"
        )

def generate_static_relationship_diagram(relationships):
    """Generate static relationship diagram using Plotly with enhanced features"""
    
    if not relationships:
        st.info("No relationships to visualize")
        return
    
    try:
        # Create network graph for visualization
        G = nx.Graph()
        
        # Add nodes (files)
        file_names = list(st.session_state.processed_data.keys())
        for file_name in file_names:
            G.add_node(file_name, size=len(st.session_state.processed_data[file_name]['dataframe'].columns))
        
        # Add edges (relationships)
        for rel in relationships:
            weight = min(rel['overlap_percentage_file1'], rel['overlap_percentage_file2'])
            G.add_edge(rel['file1'], rel['file2'], weight=weight, 
                      label=f"{rel['column1']}↔{rel['column2']}\n{rel['relationship_type']}")
        
        # Create Plotly visualization with enhanced layout
        pos = nx.spring_layout(G, k=2, iterations=100)
        
        # Edge traces
        edge_x = []
        edge_y = []
        edge_text = []
        edge_widths = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(edge[2].get('label', ''))
            edge_widths.append(max(edge[2].get('weight', 0) / 20, 1))
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888', shape='spline'),
            hoverinfo='text',
            mode='lines',
            text=edge_text * 3
        )
        
        # Node traces
        node_x = []
        node_y = []
        node_text = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}\nColumns: {G.nodes[node].get('size', 0)}")
            node_sizes.append(G.nodes[node].get('size', 10) * 3 + 20)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.split('/')[-1] for node in node_text],  # Show only filename
            textposition="middle center",
            marker=dict(
                size=node_sizes,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            )
        )
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='File Relationships Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[dict(
                               text="Static network visualization - Install pyvis for interactive version",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002)],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500)
                       )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating static visualization: {str(e)}")
        display_relationship_matrix(relationships)

def display_relationship_matrix(relationships):
    """Display relationships as a matrix with enhanced information"""
    files = list(st.session_state.processed_data.keys())
    matrix_data = []
    
    for file1 in files:
        row = {'File': file1}
        for file2 in files:
            if file1 == file2:
                row[file2] = 'Self'
            else:
                # Find relationship between file1 and file2
                rels = [r for r in relationships 
                       if (r['file1'] == file1 and r['file2'] == file2) or 
                          (r['file1'] == file2 and r['file2'] == file1)]
                if rels:
                    best_rel = max(rels, key=lambda x: x['overlap_percentage_file1'])
                    row[file2] = f"{best_rel['overlap_percentage_file1']:.1f}%"
                else:
                    row[file2] = '0%'
        matrix_data.append(row)
    
    matrix_df = pd.DataFrame(matrix_data).set_index('File')
    
    # Create styled matrix
    st.write("#### Relationship Strength Matrix")
    st.dataframe(matrix_df.style.background_gradient(cmap='Blues', axis=None), 
                use_container_width=True)