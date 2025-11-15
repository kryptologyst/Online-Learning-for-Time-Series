"""
Streamlit web interface for time series analysis.

This module provides an interactive web interface for exploring
time series analysis results and comparing different models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import logging

# Import our modules
from src.data_generator import TimeSeriesGenerator, load_config
from src.models import (
    OnlineLearningModel, ARIMAModel, ProphetModel, LSTMModel, 
    AnomalyDetector, evaluate_model
)
from src.visualization import TimeSeriesVisualizer, create_dashboard_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .model-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config_cached(config_path: str):
    """Cached configuration loading."""
    return load_config(config_path)


@st.cache_data
def generate_data_cached(config: dict, trend_type: str, seasonality_type: str, add_anomalies: bool):
    """Cached data generation."""
    generator = TimeSeriesGenerator(config['data'])
    return generator.generate_time_series(trend_type, seasonality_type, add_anomalies=add_anomalies)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'config' not in st.session_state:
        st.session_state.config = None


def create_interactive_plot(df: pd.DataFrame, predictions: dict = None, anomaly_scores: np.ndarray = None):
    """Create interactive Plotly plot."""
    fig = go.Figure()
    
    # Main time series
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['value'],
        mode='lines',
        name='Time Series',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Time:</b> %{x:.2f}<br><b>Value:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add predictions
    if predictions:
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, (model_name, pred_data) in enumerate(predictions.items()):
            if 'predictions' in pred_data:
                pred_values = pred_data['predictions']
                forecast_time = np.linspace(df['time'].max(), df['time'].max() + len(pred_values) * 0.1, len(pred_values))
                
                fig.add_trace(go.Scatter(
                    x=forecast_time,
                    y=pred_values,
                    mode='lines',
                    name=f'{model_name.title()} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    hovertemplate=f'<b>{model_name.title()} Forecast:</b> %{{y:.2f}}<extra></extra>'
                ))
    
    # Add anomalies
    if anomaly_scores is not None:
        anomaly_mask = anomaly_scores < -0.5
        if np.any(anomaly_mask):
            fig.add_trace(go.Scatter(
                x=df['time'][anomaly_mask],
                y=df['value'][anomaly_mask],
                mode='markers',
                name='Anomalies',
                marker=dict(color='#d62728', size=8, symbol='x'),
                hovertemplate='<b>Anomaly:</b> %{y:.2f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Interactive Time Series Analysis",
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        template="plotly_white",
        height=500
    )
    
    return fig


def create_model_comparison_plot(df: pd.DataFrame, model_results: dict):
    """Create model comparison plot."""
    fig = make_subplots(
        rows=len(model_results) + 1, cols=1,
        subplot_titles=['Original Data'] + [f'{name.title()} Model' for name in model_results.keys()],
        vertical_spacing=0.1
    )
    
    # Original data
    fig.add_trace(
        go.Scatter(x=df['time'], y=df['value'], mode='lines', name='Original Data', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    # Model predictions
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (model_name, results) in enumerate(model_results.items()):
        row = i + 2
        
        # Historical data
        fig.add_trace(
            go.Scatter(x=df['time'], y=df['value'], mode='lines', name='Historical Data', 
                      line=dict(color='#1f77b4', width=1), opacity=0.7),
            row=row, col=1
        )
        
        # Predictions
        if 'predictions' in results:
            predictions = results['predictions']
            forecast_time = np.linspace(df['time'].max(), df['time'].max() + len(predictions) * 0.1, len(predictions))
            
            fig.add_trace(
                go.Scatter(x=forecast_time, y=predictions, mode='lines', 
                          name=f'{model_name.title()} Forecast', 
                          line=dict(color=colors[i % len(colors)], width=2)),
                row=row, col=1
            )
    
    fig.update_layout(height=300 * (len(model_results) + 1), showlegend=False, template="plotly_white")
    return fig


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Load configuration
    try:
        config = load_config_cached("config/config.yaml")
        st.session_state.config = config
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    
    trend_type = st.sidebar.selectbox(
        "Trend Type",
        ["linear", "complex"],
        help="Type of trend component"
    )
    
    seasonality_type = st.sidebar.selectbox(
        "Seasonality Type",
        ["single", "multiple", "none"],
        help="Type of seasonality component"
    )
    
    add_anomalies = st.sidebar.checkbox(
        "Add Anomalies",
        value=False,
        help="Add anomalous points to the time series"
    )
    
    n_points = st.sidebar.slider(
        "Number of Points",
        min_value=100,
        max_value=2000,
        value=config['data']['n_points'],
        step=100
    )
    
    # Model selection
    st.sidebar.subheader("Models to Train")
    
    models_to_train = []
    if st.sidebar.checkbox("Online Learning", value=True):
        models_to_train.append("online_learning")
    if st.sidebar.checkbox("ARIMA", value=True):
        models_to_train.append("arima")
    if st.sidebar.checkbox("Prophet", value=False):
        models_to_train.append("prophet")
    if st.sidebar.checkbox("LSTM", value=False):
        models_to_train.append("lstm")
    
    # Update config with user inputs
    config['data']['n_points'] = n_points
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running analysis..."):
            try:
                # Generate data
                t, values, df = generate_data_cached(config, trend_type, seasonality_type, add_anomalies)
                
                # Initialize models
                from src.main import TimeSeriesAnalysisApp
                app = TimeSeriesAnalysisApp()
                app.config = config
                
                # Train models
                model_results = app.train_models(df, models_to_train)
                
                # Detect anomalies
                anomaly_scores, anomaly_predictions = app.detect_anomalies(df)
                
                # Store results
                st.session_state.analysis_results = {
                    'data': df,
                    'models': model_results,
                    'anomalies': {
                        'scores': anomaly_scores,
                        'predictions': anomaly_predictions
                    }
                }
                
                st.success("Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error running analysis: {e}")
                logger.error(f"Analysis error: {e}")
    
    # Display results
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        df = results['data']
        model_results = results['models']
        anomaly_scores = results['anomalies']['scores']
        
        # Main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecasting", "🚨 Anomalies", "📈 Model Comparison"])
        
        with tab1:
            st.subheader("Time Series Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Data Points", len(df))
            
            with col2:
                st.metric("Mean Value", f"{df['value'].mean():.2f}")
            
            with col3:
                st.metric("Std Deviation", f"{df['value'].std():.2f}")
            
            with col4:
                anomaly_count = np.sum(anomaly_scores < -0.5)
                st.metric("Anomalies Detected", anomaly_count)
            
            # Interactive plot
            fig = create_interactive_plot(df, model_results, anomaly_scores)
            st.plotly_chart(fig, use_container_width=True)
            
            # Data statistics
            st.subheader("Data Statistics")
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            st.subheader("Forecasting Results")
            
            if model_results:
                # Model performance metrics
                dashboard_data = create_dashboard_data(model_results)
                if not dashboard_data.empty:
                    st.subheader("Model Performance")
                    st.dataframe(dashboard_data, use_container_width=True)
                    
                    # Performance comparison chart
                    fig_perf = px.bar(
                        dashboard_data.melt(id_vars=['Model'], var_name='Metric', value_name='Value'),
                        x='Model', y='Value', color='Metric', facet_col='Metric',
                        title="Model Performance Comparison"
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                # Individual model forecasts
                for model_name, results in model_results.items():
                    if 'predictions' in results:
                        st.subheader(f"{model_name.title()} Forecast")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            fig_forecast = go.Figure()
                            
                            # Historical data
                            fig_forecast.add_trace(go.Scatter(
                                x=df['time'], y=df['value'], mode='lines',
                                name='Historical Data', line=dict(color='#1f77b4')
                            ))
                            
                            # Forecast
                            predictions = results['predictions']
                            forecast_time = np.linspace(df['time'].max(), df['time'].max() + len(predictions) * 0.1, len(predictions))
                            fig_forecast.add_trace(go.Scatter(
                                x=forecast_time, y=predictions, mode='lines',
                                name=f'{model_name.title()} Forecast', line=dict(color='#ff7f0e', dash='dash')
                            ))
                            
                            fig_forecast.update_layout(
                                title=f"{model_name.title()} Forecasting Results",
                                xaxis_title="Time",
                                yaxis_title="Value",
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        with col2:
                            if 'metrics' in results:
                                metrics = results['metrics']
                                st.metric("RMSE", f"{metrics.get('RMSE', 0):.3f}")
                                st.metric("MAE", f"{metrics.get('MAE', 0):.3f}")
                                st.metric("R²", f"{metrics.get('R2', 0):.3f}")
                                st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
            else:
                st.info("No forecasting models were trained. Please select models in the sidebar and run the analysis.")
        
        with tab3:
            st.subheader("Anomaly Detection")
            
            # Anomaly statistics
            anomaly_count = np.sum(anomaly_scores < -0.5)
            anomaly_percentage = (anomaly_count / len(anomaly_scores)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", anomaly_count)
            with col2:
                st.metric("Anomaly Rate", f"{anomaly_percentage:.2f}%")
            with col3:
                st.metric("Threshold", "-0.5")
            
            # Anomaly plot
            fig_anomaly = go.Figure()
            
            # Normal points
            normal_mask = anomaly_scores >= -0.5
            fig_anomaly.add_trace(go.Scatter(
                x=df['time'][normal_mask], y=df['value'][normal_mask],
                mode='lines', name='Normal', line=dict(color='#1f77b4')
            ))
            
            # Anomaly points
            anomaly_mask = anomaly_scores < -0.5
            if np.any(anomaly_mask):
                fig_anomaly.add_trace(go.Scatter(
                    x=df['time'][anomaly_mask], y=df['value'][anomaly_mask],
                    mode='markers', name='Anomalies', marker=dict(color='#d62728', size=8, symbol='x')
                ))
            
            fig_anomaly.update_layout(
                title="Anomaly Detection Results",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly scores plot
            fig_scores = go.Figure()
            fig_scores.add_trace(go.Scatter(
                x=df['time'], y=anomaly_scores, mode='lines',
                name='Anomaly Scores', line=dict(color='#ff7f0e')
            ))
            fig_scores.add_hline(y=-0.5, line_dash="dash", line_color="red", 
                               annotation_text="Threshold")
            
            fig_scores.update_layout(
                title="Anomaly Scores Over Time",
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                template="plotly_white"
            )
            
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with tab4:
            st.subheader("Model Comparison")
            
            if model_results:
                # Model comparison plot
                fig_comparison = create_model_comparison_plot(df, model_results)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Detailed comparison table
                if not dashboard_data.empty:
                    st.subheader("Detailed Performance Metrics")
                    st.dataframe(dashboard_data, use_container_width=True)
                    
                    # Best model
                    best_model = dashboard_data.loc[dashboard_data['R2'].idxmax(), 'Model']
                    st.success(f"🏆 Best performing model: **{best_model}** (R² = {dashboard_data['R2'].max():.3f})")
            else:
                st.info("No models to compare. Please train models in the sidebar and run the analysis.")
    
    else:
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run Analysis' to get started!")
        
        # Show example configuration
        st.subheader("Example Configuration")
        st.code("""
        # Example configuration for time series analysis
        
        Data Generation:
        - Trend Type: Linear or Complex
        - Seasonality: Single, Multiple, or None
        - Anomalies: Optional
        
        Models:
        - Online Learning: Real-time adaptation
        - ARIMA: Traditional statistical model
        - Prophet: Facebook's forecasting tool
        - LSTM: Deep learning approach
        
        Features:
        - Interactive visualizations
        - Model performance comparison
        - Anomaly detection
        - Real-time analysis
        """, language="yaml")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Time Series Analysis Dashboard | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
