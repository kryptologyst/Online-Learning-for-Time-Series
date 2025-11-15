"""
Data generation module for time series analysis.

This module provides functions to generate synthetic time series data
with various characteristics including trends, seasonality, and noise.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TimeSeriesGenerator:
    """Generator for synthetic time series data."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the time series generator.
        
        Args:
            config: Configuration dictionary containing generation parameters
        """
        self.config = config
        self.n_points = config.get('n_points', 1000)
        self.trend_strength = config.get('trend_strength', 2.0)
        self.noise_level = config.get('noise_level', 1.0)
        self.seasonality_period = config.get('seasonality_period', 50)
        self.seasonality_amplitude = config.get('seasonality_amplitude', 3.0)
        self.random_seed = config.get('random_seed', 42)
        
        np.random.seed(self.random_seed)
        logger.info(f"Initialized TimeSeriesGenerator with {self.n_points} points")
    
    def generate_linear_trend(self, t: np.ndarray) -> np.ndarray:
        """
        Generate linear trend component.
        
        Args:
            t: Time array
            
        Returns:
            Linear trend values
        """
        return self.trend_strength * t
    
    def generate_seasonality(self, t: np.ndarray) -> np.ndarray:
        """
        Generate seasonal component.
        
        Args:
            t: Time array
            
        Returns:
            Seasonal component values
        """
        return self.seasonality_amplitude * np.sin(2 * np.pi * t / self.seasonality_period)
    
    def generate_noise(self, size: int) -> np.ndarray:
        """
        Generate random noise component.
        
        Args:
            size: Number of noise points to generate
            
        Returns:
            Random noise values
        """
        return np.random.normal(0, self.noise_level, size)
    
    def generate_complex_trend(self, t: np.ndarray) -> np.ndarray:
        """
        Generate complex trend with multiple components.
        
        Args:
            t: Time array
            
        Returns:
            Complex trend values
        """
        # Linear trend
        linear = self.trend_strength * t
        
        # Quadratic component
        quadratic = 0.1 * t**2
        
        # Exponential growth component
        exponential = 0.5 * np.exp(0.1 * t)
        
        return linear + quadratic + exponential
    
    def generate_multiple_seasonalities(self, t: np.ndarray) -> np.ndarray:
        """
        Generate multiple seasonal components.
        
        Args:
            t: Time array
            
        Returns:
            Combined seasonal component
        """
        # Primary seasonality
        primary = self.seasonality_amplitude * np.sin(2 * np.pi * t / self.seasonality_period)
        
        # Secondary seasonality (higher frequency)
        secondary = 0.5 * self.seasonality_amplitude * np.sin(2 * np.pi * t / (self.seasonality_period / 2))
        
        # Tertiary seasonality (lower frequency)
        tertiary = 0.3 * self.seasonality_amplitude * np.sin(2 * np.pi * t / (self.seasonality_period * 2))
        
        return primary + secondary + tertiary
    
    def generate_time_series(
        self, 
        trend_type: str = "linear",
        seasonality_type: str = "single",
        noise_type: str = "gaussian",
        add_anomalies: bool = False,
        anomaly_probability: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate a complete time series with trend, seasonality, and noise.
        
        Args:
            trend_type: Type of trend ("linear", "complex")
            seasonality_type: Type of seasonality ("single", "multiple", "none")
            noise_type: Type of noise ("gaussian", "uniform")
            add_anomalies: Whether to add anomalous points
            anomaly_probability: Probability of anomaly occurrence
            
        Returns:
            Tuple of (time_array, values_array, dataframe)
        """
        # Generate time array
        t = np.linspace(0, 10, self.n_points)
        
        # Generate trend component
        if trend_type == "linear":
            trend = self.generate_linear_trend(t)
        elif trend_type == "complex":
            trend = self.generate_complex_trend(t)
        else:
            trend = np.zeros_like(t)
        
        # Generate seasonality component
        if seasonality_type == "single":
            seasonality = self.generate_seasonality(t)
        elif seasonality_type == "multiple":
            seasonality = self.generate_multiple_seasonalities(t)
        else:
            seasonality = np.zeros_like(t)
        
        # Generate noise component
        if noise_type == "gaussian":
            noise = self.generate_noise(self.n_points)
        elif noise_type == "uniform":
            noise = np.random.uniform(-self.noise_level, self.noise_level, self.n_points)
        else:
            noise = np.zeros(self.n_points)
        
        # Combine components
        values = trend + seasonality + noise
        
        # Add anomalies if requested
        if add_anomalies:
            anomaly_mask = np.random.random(self.n_points) < anomaly_probability
            anomaly_values = np.random.normal(0, 5 * self.noise_level, np.sum(anomaly_mask))
            values[anomaly_mask] += anomaly_values
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': t,
            'value': values,
            'trend': trend,
            'seasonality': seasonality,
            'noise': noise,
            'is_anomaly': add_anomalies and anomaly_mask if add_anomalies else np.zeros(self.n_points, dtype=bool)
        })
        
        logger.info(f"Generated time series with {self.n_points} points, "
                   f"trend_type={trend_type}, seasonality_type={seasonality_type}")
        
        return t, values, df
    
    def generate_multiple_series(self, n_series: int = 3) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple time series with different characteristics.
        
        Args:
            n_series: Number of series to generate
            
        Returns:
            Dictionary of DataFrames with different series
        """
        series_dict = {}
        
        # Linear trend with single seasonality
        _, _, df1 = self.generate_time_series("linear", "single")
        series_dict["linear_single"] = df1
        
        # Complex trend with multiple seasonalities
        _, _, df2 = self.generate_time_series("complex", "multiple")
        series_dict["complex_multiple"] = df2
        
        # Linear trend with anomalies
        _, _, df3 = self.generate_time_series("linear", "single", add_anomalies=True)
        series_dict["linear_with_anomalies"] = df3
        
        return series_dict


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import yaml
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


if __name__ == "__main__":
    # Example usage
    config = {
        'n_points': 500,
        'trend_strength': 1.5,
        'noise_level': 0.8,
        'seasonality_period': 30,
        'seasonality_amplitude': 2.0,
        'random_seed': 42
    }
    
    generator = TimeSeriesGenerator(config)
    t, values, df = generator.generate_time_series("linear", "single")
    
    print(f"Generated time series with {len(values)} points")
    print(f"Mean: {np.mean(values):.2f}, Std: {np.std(values):.2f}")
    print(f"Min: {np.min(values):.2f}, Max: {np.max(values):.2f}")
