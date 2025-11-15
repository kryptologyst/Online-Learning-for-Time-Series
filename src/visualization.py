"""
Visualization module for time series analysis.

This module provides comprehensive plotting functions for time series data,
forecasting results, and model comparisons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Comprehensive time series visualization class."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8), 
                 dpi: int = 100, colors: Optional[Dict[str, str]] = None):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
            dpi: Figure DPI
            colors: Color palette dictionary
        """
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        
        # Set default colors
        self.colors = colors or {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'tertiary': '#2ca02c',
            'anomaly': '#d62728',
            'forecast': '#9467bd'
        }
        
        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except OSError:
            logger.warning(f"Style '{self.style}' not found, using default")
            plt.style.use('default')
        
        logger.info(f"Initialized TimeSeriesVisualizer with style '{self.style}'")
    
    def plot_time_series(self, df: pd.DataFrame, title: str = "Time Series Plot", 
                        show_components: bool = True, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series with optional component decomposition.
        
        Args:
            df: DataFrame with time series data
            title: Plot title
            show_components: Whether to show trend and seasonality components
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2 if show_components else 1, 1, 
                                figsize=self.figsize, dpi=self.dpi)
        
        if show_components:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            # Main time series
            axes[0].plot(df['time'], df['value'], color=self.colors['primary'], 
                        linewidth=2, label='Time Series')
            axes[0].set_title(f"{title} - Original Data")
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Value')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Components
            axes[1].plot(df['time'], df['trend'], color=self.colors['secondary'], 
                         linewidth=2, label='Trend')
            axes[1].plot(df['time'], df['seasonality'], color=self.colors['tertiary'], 
                         linewidth=2, label='Seasonality')
            axes[1].plot(df['time'], df['noise'], color='gray', 
                         linewidth=1, alpha=0.7, label='Noise')
            axes[1].set_title("Time Series Components")
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes.plot(df['time'], df['value'], color=self.colors['primary'], 
                     linewidth=2, label='Time Series')
            axes.set_title(title)
            axes.set_xlabel('Time')
            axes.set_ylabel('Value')
            axes.legend()
            axes.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def plot_forecast(self, df: pd.DataFrame, predictions: np.ndarray, 
                     forecast_horizon: int = 10, model_name: str = "Model",
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series with forecast.
        
        Args:
            df: DataFrame with historical data
            predictions: Forecast predictions
            forecast_horizon: Number of forecast steps
            model_name: Name of the forecasting model
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Historical data
        ax.plot(df['time'], df['value'], color=self.colors['primary'], 
                linewidth=2, label='Historical Data')
        
        # Forecast
        forecast_time = np.linspace(df['time'].max(), df['time'].max() + forecast_horizon * 0.1, 
                                   forecast_horizon)
        ax.plot(forecast_time, predictions, color=self.colors['forecast'], 
                linewidth=2, linestyle='--', label=f'{model_name} Forecast')
        
        # Confidence interval (if available)
        if len(predictions) > 1:
            std_dev = np.std(predictions)
            ax.fill_between(forecast_time, 
                           predictions - 1.96 * std_dev,
                           predictions + 1.96 * std_dev,
                           color=self.colors['forecast'], alpha=0.2, 
                           label='95% Confidence Interval')
        
        ax.set_title(f"Time Series Forecast - {model_name}")
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        return fig
    
    def plot_anomalies(self, df: pd.DataFrame, anomaly_scores: np.ndarray, 
                      threshold: float = -0.5, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series with detected anomalies.
        
        Args:
            df: DataFrame with time series data
            anomaly_scores: Anomaly scores from detector
            threshold: Threshold for anomaly detection
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Time series with anomalies highlighted
        normal_mask = anomaly_scores >= threshold
        anomaly_mask = anomaly_scores < threshold
        
        ax1.plot(df['time'][normal_mask], df['value'][normal_mask], 
                color=self.colors['primary'], linewidth=2, label='Normal')
        ax1.scatter(df['time'][anomaly_mask], df['value'][anomaly_mask], 
                   color=self.colors['anomaly'], s=50, label='Anomalies', zorder=5)
        ax1.set_title("Time Series with Anomaly Detection")
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Anomaly scores
        ax2.plot(df['time'], anomaly_scores, color=self.colors['secondary'], 
                linewidth=1, label='Anomaly Score')
        ax2.axhline(y=threshold, color=self.colors['anomaly'], 
                   linestyle='--', label=f'Threshold ({threshold})')
        ax2.fill_between(df['time'], threshold, anomaly_scores.min(), 
                         where=(anomaly_scores < threshold), 
                         color=self.colors['anomaly'], alpha=0.3, 
                         label='Anomaly Region')
        ax2.set_title("Anomaly Scores")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Anomaly Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Anomaly plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(self, df: pd.DataFrame, 
                            model_results: Dict[str, Dict[str, Any]], 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of multiple models.
        
        Args:
            df: DataFrame with time series data
            model_results: Dictionary with model results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_models = len(model_results)
        fig, axes = plt.subplots(n_models + 1, 1, figsize=(self.figsize[0], self.figsize[1] * (n_models + 1) / 2), 
                                dpi=self.dpi)
        
        if n_models == 1:
            axes = [axes]
        
        # Original data
        axes[0].plot(df['time'], df['value'], color=self.colors['primary'], 
                     linewidth=2, label='Original Data')
        axes[0].set_title("Original Time Series")
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Model predictions
        colors = [self.colors['secondary'], self.colors['tertiary'], self.colors['forecast']]
        for i, (model_name, results) in enumerate(model_results.items()):
            ax = axes[i + 1]
            
            # Historical data
            ax.plot(df['time'], df['value'], color=self.colors['primary'], 
                   linewidth=1, alpha=0.7, label='Original Data')
            
            # Predictions
            if 'predictions' in results:
                predictions = results['predictions']
                forecast_time = np.linspace(df['time'].max(), df['time'].max() + len(predictions) * 0.1, 
                                           len(predictions))
                ax.plot(forecast_time, predictions, color=colors[i % len(colors)], 
                       linewidth=2, label=f'{model_name} Forecast')
            
            # Metrics
            metrics_text = ""
            if 'metrics' in results:
                metrics = results['metrics']
                metrics_text = f"RMSE: {metrics.get('RMSE', 0):.3f}, R²: {metrics.get('R2', 0):.3f}"
            
            ax.set_title(f"{model_name} - {metrics_text}")
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_interactive_time_series(self, df: pd.DataFrame, 
                                   predictions: Optional[np.ndarray] = None,
                                   anomaly_scores: Optional[np.ndarray] = None) -> go.Figure:
        """
        Create interactive Plotly time series plot.
        
        Args:
            df: DataFrame with time series data
            predictions: Optional forecast predictions
            anomaly_scores: Optional anomaly scores
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Main time series
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['value'],
            mode='lines',
            name='Time Series',
            line=dict(color=self.colors['primary'], width=2)
        ))
        
        # Add predictions if available
        if predictions is not None:
            forecast_time = np.linspace(df['time'].max(), df['time'].max() + len(predictions) * 0.1, 
                                       len(predictions))
            fig.add_trace(go.Scatter(
                x=forecast_time,
                y=predictions,
                mode='lines',
                name='Forecast',
                line=dict(color=self.colors['forecast'], width=2, dash='dash')
            ))
        
        # Add anomalies if available
        if anomaly_scores is not None:
            anomaly_mask = anomaly_scores < -0.5
            if np.any(anomaly_mask):
                fig.add_trace(go.Scatter(
                    x=df['time'][anomaly_mask],
                    y=df['value'][anomaly_mask],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color=self.colors['anomaly'], size=8)
                ))
        
        fig.update_layout(
            title="Interactive Time Series Analysis",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template="plotly_white"
        )
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, 
                              columns: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix for time series components.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to include in correlation matrix
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if columns is None:
            columns = ['value', 'trend', 'seasonality', 'noise']
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        
        ax.set_title("Time Series Components Correlation Matrix")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        return fig
    
    def plot_distribution(self, df: pd.DataFrame, 
                         columns: Optional[List[str]] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot distribution of time series components.
        
        Args:
            df: DataFrame with time series data
            columns: Columns to plot distributions for
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if columns is None:
            columns = ['value', 'trend', 'seasonality', 'noise']
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(self.figsize[0], self.figsize[1] // 2), 
                               dpi=self.dpi)
        
        if n_cols == 1:
            axes = [axes]
        
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['tertiary'], self.colors['forecast']]
        
        for i, col in enumerate(columns):
            if col in df.columns:
                axes[i].hist(df[col], bins=30, alpha=0.7, color=colors[i % len(colors)])
                axes[i].set_title(f"{col.title()} Distribution")
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")
        
        return fig


def create_dashboard_data(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Create data for model comparison dashboard.
    
    Args:
        model_results: Dictionary with model results
        
    Returns:
        DataFrame with model comparison data
    """
    dashboard_data = []
    
    for model_name, results in model_results.items():
        if 'metrics' in results:
            metrics = results['metrics']
            dashboard_data.append({
                'Model': model_name,
                'RMSE': metrics.get('RMSE', 0),
                'MAE': metrics.get('MAE', 0),
                'R2': metrics.get('R2', 0),
                'MAPE': metrics.get('MAPE', 0)
            })
    
    return pd.DataFrame(dashboard_data)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    trend = 2 * t
    seasonality = 3 * np.sin(2 * np.pi * t / 5)
    noise = np.random.randn(100)
    values = trend + seasonality + noise
    
    df = pd.DataFrame({
        'time': t,
        'value': values,
        'trend': trend,
        'seasonality': seasonality,
        'noise': noise
    })
    
    # Create visualizer
    viz = TimeSeriesVisualizer()
    
    # Plot time series
    fig1 = viz.plot_time_series(df, "Example Time Series")
    plt.show()
    
    # Plot distribution
    fig2 = viz.plot_distribution(df)
    plt.show()
    
    print("Visualization examples completed successfully!")
