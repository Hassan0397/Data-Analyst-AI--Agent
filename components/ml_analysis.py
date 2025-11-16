import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.ml_pipeline import MLPipeline

class MLAnalysis:
    def __init__(self):
        self.ml_pipeline = MLPipeline()
        self.models = {}
        self.results = {}
    
    def render(self, df):
        """Render machine learning analysis interface"""
        st.markdown('<div class="section-header">🤖 Machine Learning Analysis</div>', unsafe_allow_html=True)
        
        if df is None:
            st.warning("Please upload data to perform ML analysis")
            return
        
        tabs = st.tabs([
            "🎯 Problem Setup",
            "🔧 Model Training", 
            "📊 Model Evaluation",
            "📈 Feature Importance",
            "🚀 Predictions"
        ])
        
        with tabs[0]:
            self._render_problem_setup(df)
        
        with tabs[1]:
            self._render_model_training(df)
        
        with tabs[2]:
            self._render_model_evaluation(df)
        
        with tabs[3]:
            self._render_feature_importance(df)
        
        with tabs[4]:
            self._render_predictions(df)
    
    def _render_problem_setup(self, df):
        """Render ML problem setup"""
        st.subheader("🎯 Machine Learning Problem Setup")
        
        # Problem type selection
        problem_type = st.radio(
            "Select Problem Type:",
            ["Classification", "Regression", "Clustering"],
            horizontal=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target variable selection
            if problem_type in ["Classification", "Regression"]:
                target_var = st.selectbox(
                    "Select Target Variable:",
                    df.columns,
                    key="target_var"
                )
                
                # Feature selection
                feature_vars = st.multiselect(
                    "Select Feature Variables:",
                    [col for col in df.columns if col != target_var],
                    default=[col for col in df.select_dtypes(include=[np.number]).columns 
                            if col != target_var][:5]
                )
            
            elif problem_type == "Clustering":
                feature_vars = st.multiselect(
                    "Select Variables for Clustering:",
                    df.select_dtypes(include=[np.number]).columns,
                    default=df.select_dtypes(include=[np.number]).columns[:5]
                )
        
        with col2:
            # Training parameters
            test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5)
            random_state = st.number_input("Random State", 42, key="random_state")
            
            if problem_type in ["Classification", "Regression"]:
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        
        # Data preview
        if problem_type in ["Classification", "Regression"] and target_var and feature_vars:
            st.subheader("📋 Data Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Target Variable Distribution**")
                if problem_type == "Classification":
                    target_dist = df[target_var].value_counts()
                    fig = px.bar(x=target_dist.index, y=target_dist.values,
                               title=f"Distribution of {target_var}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.histogram(df, x=target_var, 
                                     title=f"Distribution of {target_var}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Feature Statistics**")
                feature_stats = df[feature_vars].describe()
                st.dataframe(feature_stats, use_container_width=True)
        
        # Store setup in session state
        if st.button("💾 Save Problem Setup", type="primary"):
            st.session_state.ml_setup = {
                'problem_type': problem_type,
                'target_var': target_var if problem_type != "Clustering" else None,
                'feature_vars': feature_vars,
                'test_size': test_size / 100,
                'random_state': random_state,
                'cv_folds': cv_folds if problem_type != "Clustering" else None
            }
            st.success("Problem setup saved! Proceed to Model Training.")
    
    def _render_model_training(self, df):
        """Render model training interface"""
        st.subheader("🔧 Model Training")
        
        if 'ml_setup' not in st.session_state:
            st.warning("Please complete the Problem Setup first.")
            return
        
        setup = st.session_state.ml_setup
        problem_type = setup['problem_type']
        
        # Model selection based on problem type
        if problem_type == "Classification":
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(),
                "Support Vector Machine": SVC()
            }
        elif problem_type == "Regression":
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Support Vector Regression": SVR()
            }
        else:  # Clustering
            models = {
                "K-Means": "kmeans",
                "DBSCAN": "dbscan",
                "Hierarchical": "hierarchical"
            }
        
        # Model selection
        selected_models = st.multiselect(
            "Select Models to Train:",
            list(models.keys()),
            default=list(models.keys())[:2]
        )
        
        # Hyperparameter tuning
        st.subheader("⚙️ Hyperparameter Configuration")
        
        param_config = {}
        for model_name in selected_models:
            with st.expander(f"Parameters for {model_name}"):
                if model_name == "Random Forest":
                    n_estimators = st.slider("Number of Trees", 50, 500, 100, 50, key=f"rf_trees")
                    max_depth = st.slider("Max Depth", 3, 20, 10, 1, key=f"rf_depth")
                    param_config[model_name] = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth
                    }
                elif model_name == "Logistic Regression":
                    C = st.slider("Regularization (C)", 0.01, 10.0, 1.0, 0.1, key=f"lr_c")
                    param_config[model_name] = {'C': C}
        
        # Train models
        if st.button("🚀 Train Selected Models", type="primary"):
            if problem_type in ["Classification", "Regression"]:
                self._train_supervised_models(df, setup, selected_models, models, param_config)
            else:
                self._train_unsupervised_models(df, setup, selected_models, models)
    
    def _train_supervised_models(self, df, setup, selected_models, models, param_config):
        """Train supervised learning models"""
        X = df[setup['feature_vars']]
        y = df[setup['target_var']]
        
        # Handle categorical variables
        X = self._preprocess_features(X)
        
        # Encode target for classification
        if setup['problem_type'] == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.label_encoder = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=setup['test_size'],
            random_state=setup['random_state'],
            stratify=y if setup['problem_type'] == "Classification" else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.session_state.scaler = scaler
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        # Train models
        progress_bar = st.progress(0)
        self.results = {}
        
        for i, model_name in enumerate(selected_models):
            progress = (i / len(selected_models))
            progress_bar.progress(progress)
            
            with st.spinner(f"Training {model_name}..."):
                # Get model with parameters
                model = models[model_name]
                if model_name in param_config:
                    model.set_params(**param_config[model_name])
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                if setup['problem_type'] == "Classification":
                    accuracy = model.score(X_test_scaled, y_test)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                              cv=setup['cv_folds'])
                    
                    self.results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                
                else:  # Regression
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                              cv=setup['cv_folds'], scoring='r2')
                    
                    self.results[model_name] = {
                        'model': model,
                        'r2_score': r2,
                        'mse': mse,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred
                    }
        
        progress_bar.progress(1.0)
        st.success("✅ All models trained successfully!")
        
        # Display results
        self._display_training_results(setup['problem_type'])
    
    def _train_unsupervised_models(self, df, setup, selected_models, models):
        """Train unsupervised learning models"""
        X = df[setup['feature_vars']]
        X_scaled = StandardScaler().fit_transform(X)
        
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        self.results = {}
        
        for model_name in selected_models:
            with st.spinner(f"Training {model_name}..."):
                if model_name == "K-Means":
                    # Find optimal k
                    inertia = []
                    silhouette_scores = []
                    k_range = range(2, 11)
                    
                    for k in k_range:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        labels = kmeans.fit_predict(X_scaled)
                        inertia.append(kmeans.inertia_)
                        silhouette_scores.append(silhouette_score(X_scaled, labels))
                    
                    # Use elbow method to select k
                    optimal_k = self._find_optimal_k(inertia, k_range)
                    
                    # Train with optimal k
                    final_model = KMeans(n_clusters=optimal_k, random_state=42)
                    labels = final_model.fit_predict(X_scaled)
                    
                    self.results[model_name] = {
                        'model': final_model,
                        'labels': labels,
                        'inertia': final_model.inertia_,
                        'silhouette_score': silhouette_score(X_scaled, labels),
                        'optimal_k': optimal_k
                    }
        
        st.success("✅ Clustering completed!")
        self._display_clustering_results(df, X_scaled)
    
    def _find_optimal_k(self, inertia, k_range):
        """Find optimal k using elbow method"""
        # Simple elbow detection (you can make this more sophisticated)
        differences = [inertia[i-1] - inertia[i] for i in range(1, len(inertia))]
        optimal_k = k_range[differences.index(max(differences)) + 1]
        return optimal_k
    
    def _preprocess_features(self, X):
        """Preprocess feature variables"""
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        X_processed = X[numerical_cols].copy()
        
        for col in categorical_cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed, dummies], axis=1)
        
        return X_processed
    
    def _display_training_results(self, problem_type):
        """Display training results"""
        st.subheader("📊 Training Results")
        
        # Results table
        results_data = []
        for model_name, result in self.results.items():
            if problem_type == "Classification":
                results_data.append({
                    'Model': model_name,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'CV Score (Mean)': f"{result['cv_mean']:.4f}",
                    'CV Score (Std)': f"{result['cv_std']:.4f}"
                })
            else:  # Regression
                results_data.append({
                    'Model': model_name,
                    'R² Score': f"{result['r2_score']:.4f}",
                    'MSE': f"{result['mse']:.4f}",
                    'CV R² (Mean)': f"{result['cv_mean']:.4f}",
                    'CV R² (Std)': f"{result['cv_std']:.4f}"
                })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        # Best model
        if problem_type == "Classification":
            best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        else:
            best_model = max(self.results.items(), key=lambda x: x[1]['r2_score'])
        
        st.success(f"🎯 Best Model: **{best_model[0]}** "
                  f"({'Accuracy' if problem_type == 'Classification' else 'R² Score'}: "
                  f"{best_model[1]['accuracy' if problem_type == 'Classification' else 'r2_score']:.4f})")
        
        # Store best model
        st.session_state.best_model = best_model[1]['model']
        st.session_state.all_results = self.results
    
    def _display_clustering_results(self, df, X_scaled):
        """Display clustering results"""
        st.subheader("📊 Clustering Results")
        
        for model_name, result in self.results.items():
            with st.expander(f"{model_name} Results"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Number of Clusters", result.get('optimal_k', 'N/A'))
                    st.metric("Silhouette Score", f"{result['silhouette_score']:.4f}")
                    if 'inertia' in result:
                        st.metric("Inertia", f"{result['inertia']:.2f}")
                
                with col2:
                    # Cluster visualization
                    if X_scaled.shape[1] >= 2:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig = px.scatter(
                            x=X_pca[:, 0], y=X_pca[:, 1],
                            color=result['labels'],
                            title=f"{model_name} Clustering (PCA Visualization)",
                            labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    def _render_model_evaluation(self, df):
        """Render model evaluation interface"""
        st.subheader("📊 Model Evaluation")
        
        if 'all_results' not in st.session_state:
            st.warning("Please train models first.")
            return
        
        setup = st.session_state.ml_setup
        results = st.session_state.all_results
        
        if setup['problem_type'] == "Classification":
            self._evaluate_classification_models(results)
        elif setup['problem_type'] == "Regression":
            self._evaluate_regression_models(results)
    
    def _evaluate_classification_models(self, results):
        """Evaluate classification models"""
        st.subheader("📈 Classification Model Evaluation")
        
        # Model comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_scores = [results[name]['cv_mean'] for name in model_names]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Test Accuracy', x=model_names, y=accuracies))
        fig.add_trace(go.Bar(name='CV Score', x=model_names, y=cv_scores))
        fig.update_layout(title="Model Performance Comparison", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed evaluation for each model
        selected_model = st.selectbox("Select model for detailed evaluation:", model_names)
        
        if selected_model:
            result = results[selected_model]
            y_test = st.session_state.y_test
            y_pred = result['predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, 
                              title=f"Confusion Matrix - {selected_model}",
                              labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
    
    def _evaluate_regression_models(self, results):
        """Evaluate regression models"""
        st.subheader("📈 Regression Model Evaluation")
        
        # Model comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2_score'] for name in model_names]
        mse_scores = [results[name]['mse'] for name in model_names]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=("R² Scores", "MSE Scores"))
        
        fig.add_trace(go.Bar(name='R² Score', x=model_names, y=r2_scores), 1, 1)
        fig.add_trace(go.Bar(name='MSE', x=model_names, y=mse_scores), 1, 2)
        
        fig.update_layout(title="Regression Model Performance Comparison")
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction vs Actual plot
        selected_model = st.selectbox("Select model for detailed evaluation:", model_names)
        
        if selected_model:
            result = results[selected_model]
            y_test = st.session_state.y_test
            y_pred = result['predictions']
            
            fig = px.scatter(x=y_test, y=y_pred, 
                           labels={'x': 'Actual', 'y': 'Predicted'},
                           title=f"Actual vs Predicted - {selected_model}")
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash', color='red')))
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_importance(self, df):
        """Render feature importance analysis"""
        st.subheader("📈 Feature Importance Analysis")
        
        if 'best_model' not in st.session_state:
            st.warning("Please train models first to see feature importance.")
            return
        
        best_model = st.session_state.best_model
        setup = st.session_state.ml_setup
        
        # Get feature names after preprocessing
        X_processed = self._preprocess_features(df[setup['feature_vars']])
        feature_names = X_processed.columns.tolist()
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig = px.bar(feature_imp_df.head(20), 
                        x='Importance', y='Feature',
                        title="Top 20 Feature Importances",
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
            
            # Display feature importance table
            st.dataframe(feature_imp_df, use_container_width=True)
        
        else:
            st.info("Feature importance is not available for the selected model.")
    
    def _render_predictions(self, df):
        """Render prediction interface"""
        st.subheader("🚀 Make Predictions")
        
        if 'best_model' not in st.session_state:
            st.warning("Please train models first to make predictions.")
            return
        
        setup = st.session_state.ml_setup
        
        # Prediction mode
        prediction_mode = st.radio(
            "Prediction Mode:",
            ["Use Test Data", "Manual Input", "Upload New Data"],
            horizontal=True
        )
        
        if prediction_mode == "Use Test Data":
            self._predict_on_test_data()
        elif prediction_mode == "Manual Input":
            self._predict_manual_input(df, setup)
        elif prediction_mode == "Upload New Data":
            self._predict_upload_data(df, setup)
    
    def _predict_on_test_data(self):
        """Make predictions on test data"""
        if 'X_test' not in st.session_state:
            st.warning("Test data not available.")
            return
        
        X_test = st.session_state.X_test
        best_model = st.session_state.best_model
        scaler = st.session_state.scaler
        
        X_test_scaled = scaler.transform(X_test)
        predictions = best_model.predict(X_test_scaled)
        
        # Display predictions
        results_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': predictions
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Download predictions
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="model_predictions.csv",
            mime="text/csv"
        )
    
    def _predict_manual_input(self, df, setup):
        """Make predictions on manual input"""
        st.subheader("🔮 Manual Prediction")
        
        input_data = {}
        col1, col2 = st.columns(2)
        
        features = setup['feature_vars']
        for i, feature in enumerate(features):
            col = col1 if i % 2 == 0 else col2
            
            with col:
                if df[feature].dtype in [np.number]:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default_val = float(df[feature].median())
                    
                    input_data[feature] = st.number_input(
                        feature, min_val, max_val, default_val
                    )
                else:
                    unique_vals = df[feature].unique()
                    input_data[feature] = st.selectbox(feature, unique_vals)
        
        if st.button("🎯 Predict", type="primary"):
            # Preprocess input
            input_df = pd.DataFrame([input_data])
            input_processed = self._preprocess_features(input_df)
            
            # Ensure all columns are present
            X_processed = self._preprocess_features(df[setup['feature_vars']])
            for col in X_processed.columns:
                if col not in input_processed.columns:
                    input_processed[col] = 0
            
            input_processed = input_processed[X_processed.columns]
            
            # Scale and predict
            scaler = st.session_state.scaler
            input_scaled = scaler.transform(input_processed)
            
            best_model = st.session_state.best_model
            prediction = best_model.predict(input_scaled)
            
            if setup['problem_type'] == "Classification":
                le = st.session_state.label_encoder
                prediction_label = le.inverse_transform(prediction)[0]
                st.success(f"🎯 Prediction: **{prediction_label}**")
                
                if hasattr(best_model, 'predict_proba'):
                    probabilities = best_model.predict_proba(input_scaled)[0]
                    prob_df = pd.DataFrame({
                        'Class': le.classes_,
                        'Probability': probabilities
                    }).sort_values('Probability', ascending=False)
                    
                    fig = px.bar(prob_df, x='Class', y='Probability',
                               title="Prediction Probabilities")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success(f"🎯 Prediction: **{prediction[0]:.4f}**")
    
    def _predict_upload_data(self, df, setup):
        """Make predictions on uploaded data"""
        st.subheader("📤 Predict on New Data")
        
        uploaded_file = st.file_uploader(
            "Upload new data for predictions",
            type=['csv', 'xlsx'],
            key="prediction_file"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    new_data = pd.read_csv(uploaded_file)
                else:
                    new_data = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded data with {new_data.shape[0]} rows and {new_data.shape[1]} columns")
                
                # Check if required features are present
                missing_features = set(setup['feature_vars']) - set(new_data.columns)
                if missing_features:
                    st.error(f"❌ Missing features: {list(missing_features)}")
                    return
                
                # Make predictions
                X_new = new_data[setup['feature_vars']]
                X_processed = self._preprocess_features(X_new)
                
                # Ensure correct column order
                X_train_processed = self._preprocess_features(df[setup['feature_vars']])
                for col in X_train_processed.columns:
                    if col not in X_processed.columns:
                        X_processed[col] = 0
                
                X_processed = X_processed[X_train_processed.columns]
                
                # Scale and predict
                scaler = st.session_state.scaler
                X_scaled = scaler.transform(X_processed)
                
                best_model = st.session_state.best_model
                predictions = best_model.predict(X_scaled)
                
                # Add predictions to data
                results_data = new_data.copy()
                if setup['problem_type'] == "Classification":
                    le = st.session_state.label_encoder
                    results_data['Prediction'] = le.inverse_transform(predictions)
                else:
                    results_data['Prediction'] = predictions
                
                st.subheader("📊 Prediction Results")
                st.dataframe(results_data, use_container_width=True)
                
                # Download results
                csv = results_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Predictions",
                    data=csv,
                    file_name="new_data_predictions.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")