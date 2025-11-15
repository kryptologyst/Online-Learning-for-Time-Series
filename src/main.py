"""
Main application for time series analysis.

This module provides the main interface for running time series analysis
with various models and visualizations.
"""

import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

# Import our modules
from src.data_generator import TimeSeriesGenerator, load_config
from src.models import (
    OnlineLearningModel, ARIMAModel, ProphetModel, LSTMModel, 
    AnomalyDetector, evaluate_model
)
from src.visualization import TimeSeriesVisualizer, create_dashboard_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/time_series_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class TimeSeriesAnalysisApp:
    """Main application class for time series analysis."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the application.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.data_config = self.config['data']
        self.model_config = self.config['models']
        self.viz_config = self.config['visualization']
        
        # Initialize components
        self.data_generator = TimeSeriesGenerator(self.data_config)
        self.visualizer = TimeSeriesVisualizer(
            style=self.viz_config['style'],
            figsize=tuple(self.viz_config['figure_size']),
            dpi=self.viz_config['dpi'],
            colors=self.viz_config['colors']
        )
        
        # Initialize models
        self.models = {}
        self.results = {}
        
        logger.info("TimeSeriesAnalysisApp initialized successfully")
    
    def generate_data(self, trend_type: str = "linear", 
                     seasonality_type: str = "single",
                     add_anomalies: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate time series data.
        
        Args:
            trend_type: Type of trend
            seasonality_type: Type of seasonality
            add_anomalies: Whether to add anomalies
            
        Returns:
            Tuple of (time_array, values_array, dataframe)
        """
        logger.info(f"Generating data: trend={trend_type}, seasonality={seasonality_type}")
        return self.data_generator.generate_time_series(
            trend_type=trend_type,
            seasonality_type=seasonality_type,
            add_anomalies=add_anomalies
        )
    
    def train_models(self, df: pd.DataFrame, models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Train multiple models on the data.
        
        Args:
            df: DataFrame with time series data
            models_to_train: List of models to train
            
        Returns:
            Dictionary with model results
        """
        if models_to_train is None:
            models_to_train = ['online_learning', 'arima', 'prophet', 'lstm']
        
        logger.info(f"Training models: {models_to_train}")
        
        # Prepare data
        X = df['time'].values
        y = df['value'].values
        
        # Split data for training and testing
        split_idx = int(0.8 * len(y))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = {}
        
        # Train Online Learning Model
        if 'online_learning' in models_to_train:
            try:
                logger.info("Training Online Learning model...")
                online_config = self.model_config['online_learning']
                online_model = OnlineLearningModel(
                    window_size=online_config['window_size'],
                    learning_rate=online_config['learning_rate']
                )
                
                # Online prediction on training data
                online_predictions = online_model.online_predict(y_train)
                
                # Evaluate on test data
                test_predictions = []
                for i in range(len(online_predictions), len(y_test)):
                    if i >= online_model.window_size:
                        X_test_window = y_test[i-online_model.window_size:i].reshape(-1, 1)
                        y_test_window = y_test[i-online_model.window_size:i]
                        online_model.fit(X_test_window, y_test_window)
                        pred = online_model.predict(y_test[i].reshape(-1, 1))
                        test_predictions.append(pred[0])
                
                if test_predictions:
                    metrics = evaluate_model(y_test[-len(test_predictions):], np.array(test_predictions))
                else:
                    metrics = {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0}
                
                results['online_learning'] = {
                    'model': online_model,
                    'predictions': online_predictions,
                    'metrics': metrics
                }
                
                logger.info(f"Online Learning - RMSE: {metrics['RMSE']:.3f}, R²: {metrics['R2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training Online Learning model: {e}")
        
        # Train ARIMA Model
        if 'arima' in models_to_train:
            try:
                logger.info("Training ARIMA model...")
                arima_config = self.model_config['arima']
                arima_model = ARIMAModel(
                    order=tuple(arima_config['order']),
                    seasonal_order=tuple(arima_config['seasonal_order']) if arima_config.get('seasonal_order') else None
                )
                
                arima_model.fit(X_train, y_train)
                arima_predictions = arima_model.predict(X_test, steps=len(y_test))
                
                metrics = evaluate_model(y_test, arima_predictions)
                
                results['arima'] = {
                    'model': arima_model,
                    'predictions': arima_predictions,
                    'metrics': metrics
                }
                
                logger.info(f"ARIMA - RMSE: {metrics['RMSE']:.3f}, R²: {metrics['R2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training ARIMA model: {e}")
        
        # Train Prophet Model
        if 'prophet' in models_to_train:
            try:
                logger.info("Training Prophet model...")
                prophet_config = self.model_config['prophet']
                prophet_model = ProphetModel(**prophet_config)
                
                prophet_model.fit(X_train, y_train)
                prophet_predictions = prophet_model.predict(X_test, steps=len(y_test))
                
                metrics = evaluate_model(y_test, prophet_predictions)
                
                results['prophet'] = {
                    'model': prophet_model,
                    'predictions': prophet_predictions,
                    'metrics': metrics
                }
                
                logger.info(f"Prophet - RMSE: {metrics['RMSE']:.3f}, R²: {metrics['R2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training Prophet model: {e}")
        
        # Train LSTM Model
        if 'lstm' in models_to_train:
            try:
                logger.info("Training LSTM model...")
                lstm_config = self.model_config['lstm']
                lstm_model = LSTMModel(
                    sequence_length=lstm_config['sequence_length'],
                    hidden_size=lstm_config['hidden_size'],
                    num_layers=lstm_config['num_layers'],
                    dropout=lstm_config['dropout'],
                    epochs=lstm_config['epochs'],
                    batch_size=lstm_config['batch_size']
                )
                
                lstm_model.fit(X_train, y_train)
                lstm_predictions = lstm_model.predict(y_test, steps=len(y_test))
                
                metrics = evaluate_model(y_test, lstm_predictions)
                
                results['lstm'] = {
                    'model': lstm_model,
                    'predictions': lstm_predictions,
                    'metrics': metrics
                }
                
                logger.info(f"LSTM - RMSE: {metrics['RMSE']:.3f}, R²: {metrics['R2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training LSTM model: {e}")
        
        self.results = results
        return results
    
    def detect_anomalies(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in the time series.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            Tuple of (anomaly_scores, anomaly_predictions)
        """
        logger.info("Detecting anomalies...")
        
        # Prepare features for anomaly detection
        features = df[['value', 'trend', 'seasonality', 'noise']].values
        
        # Initialize and fit anomaly detector
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(features)
        
        # Get anomaly scores and predictions
        anomaly_scores = detector.score_samples(features)
        anomaly_predictions = detector.predict(features)
        
        logger.info(f"Detected {np.sum(anomaly_predictions == -1)} anomalies out of {len(anomaly_predictions)} points")
        
        return anomaly_scores, anomaly_predictions
    
    def create_visualizations(self, df: pd.DataFrame, save_plots: bool = True) -> Dict[str, Any]:
        """
        Create comprehensive visualizations.
        
        Args:
            df: DataFrame with time series data
            save_plots: Whether to save plots to files
            
        Returns:
            Dictionary with plot objects
        """
        logger.info("Creating visualizations...")
        
        plots = {}
        
        # Time series plot
        plots['time_series'] = self.visualizer.plot_time_series(
            df, "Generated Time Series", 
            save_path="plots/time_series.png" if save_plots else None
        )
        
        # Distribution plot
        plots['distribution'] = self.visualizer.plot_distribution(
            df, 
            save_path="plots/distribution.png" if save_plots else None
        )
        
        # Correlation matrix
        plots['correlation'] = self.visualizer.plot_correlation_matrix(
            df, 
            save_path="plots/correlation.png" if save_plots else None
        )
        
        # Model comparison plot
        if self.results:
            plots['model_comparison'] = self.visualizer.plot_model_comparison(
                df, self.results,
                save_path="plots/model_comparison.png" if save_plots else None
            )
        
        # Anomaly detection plot
        if 'is_anomaly' in df.columns:
            anomaly_scores, _ = self.detect_anomalies(df)
            plots['anomalies'] = self.visualizer.plot_anomalies(
                df, anomaly_scores,
                save_path="plots/anomalies.png" if save_plots else None
            )
        
        logger.info(f"Created {len(plots)} visualizations")
        return plots
    
    def run_complete_analysis(self, trend_type: str = "linear", 
                            seasonality_type: str = "single",
                            add_anomalies: bool = False,
                            models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Run complete time series analysis pipeline.
        
        Args:
            trend_type: Type of trend
            seasonality_type: Type of seasonality
            add_anomalies: Whether to add anomalies
            models_to_train: List of models to train
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting complete time series analysis...")
        
        # Generate data
        t, values, df = self.generate_data(trend_type, seasonality_type, add_anomalies)
        
        # Train models
        model_results = self.train_models(df, models_to_train)
        
        # Detect anomalies
        anomaly_scores, anomaly_predictions = self.detect_anomalies(df)
        
        # Create visualizations
        plots = self.create_visualizations(df)
        
        # Create dashboard data
        dashboard_data = create_dashboard_data(model_results)
        
        # Compile results
        analysis_results = {
            'data': {
                'time': t,
                'values': values,
                'dataframe': df
            },
            'models': model_results,
            'anomalies': {
                'scores': anomaly_scores,
                'predictions': anomaly_predictions
            },
            'visualizations': plots,
            'dashboard_data': dashboard_data,
            'config': self.config
        }
        
        logger.info("Complete analysis finished successfully")
        return analysis_results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "results") -> None:
        """
        Save analysis results to files.
        
        Args:
            results: Analysis results dictionary
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Saving results to {output_dir}")
        
        # Save data
        results['data']['dataframe'].to_csv(output_path / "time_series_data.csv", index=False)
        
        # Save model metrics
        if results['dashboard_data'] is not None and not results['dashboard_data'].empty:
            results['dashboard_data'].to_csv(output_path / "model_metrics.csv", index=False)
        
        # Save anomaly results
        anomaly_df = pd.DataFrame({
            'time': results['data']['time'],
            'value': results['data']['values'],
            'anomaly_score': results['anomalies']['scores'],
            'is_anomaly': results['anomalies']['predictions'] == -1
        })
        anomaly_df.to_csv(output_path / "anomaly_results.csv", index=False)
        
        # Save configuration
        with open(output_path / "config.yaml", 'w') as f:
            yaml.dump(results['config'], f, default_flow_style=False)
        
        logger.info("Results saved successfully")


def main():
    """Main function to run the analysis."""
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Initialize application
    app = TimeSeriesAnalysisApp()
    
    # Run complete analysis
    results = app.run_complete_analysis(
        trend_type="linear",
        seasonality_type="single",
        add_anomalies=True,
        models_to_train=['online_learning', 'arima', 'prophet']
    )
    
    # Save results
    app.save_results(results)
    
    # Display summary
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"Data points: {len(results['data']['values'])}")
    print(f"Models trained: {len(results['models'])}")
    print(f"Anomalies detected: {np.sum(results['anomalies']['predictions'] == -1)}")
    
    if results['dashboard_data'] is not None and not results['dashboard_data'].empty:
        print("\nModel Performance:")
        print(results['dashboard_data'].to_string(index=False))
    
    print(f"\nResults saved to: results/")
    print(f"Plots saved to: plots/")
    print(f"Logs saved to: logs/")


if __name__ == "__main__":
    main()
