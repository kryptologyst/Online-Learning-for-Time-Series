"""
Time series forecasting models module.

This module provides implementations of various time series forecasting methods
including online learning, ARIMA, Prophet, LSTM, and anomaly detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

# Traditional models
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# Deep learning models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Prophet (if available)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

logger = logging.getLogger(__name__)


class BaseTimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass


class OnlineLearningModel(BaseTimeSeriesModel):
    """Online learning model using SGD regressor."""
    
    def __init__(self, window_size: int = 10, learning_rate: float = 0.01, **kwargs):
        """
        Initialize online learning model.
        
        Args:
            window_size: Size of the sliding window for features
            learning_rate: Learning rate for SGD
            **kwargs: Additional parameters for SGDRegressor
        """
        self.window_size = window_size
        self.model = SGDRegressor(
            max_iter=10, 
            warm_start=True, 
            penalty=None,
            learning_rate='constant',
            eta0=learning_rate,
            **kwargs
        )
        self.is_fitted = False
        self.predictions_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model incrementally.
        
        Args:
            X: Feature matrix
            y: Target values
        """
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        self.predictions_history.extend(predictions)
        return predictions
    
    def online_predict(self, time_series: np.ndarray) -> List[float]:
        """
        Perform online prediction on a time series.
        
        Args:
            time_series: Input time series
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for i in range(self.window_size, len(time_series)):
            # Create features from sliding window
            X = time_series[i-self.window_size:i].reshape(-1, 1)
            y = time_series[i-self.window_size:i]
            
            # Fit incrementally
            self.fit(X, y)
            
            # Predict next value
            next_pred = self.predict(time_series[i].reshape(-1, 1))
            predictions.append(next_pred[0])
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'Online Learning (SGD)',
            'window_size': self.window_size,
            'learning_rate': self.model.eta0,
            'is_fitted': self.is_fitted,
            'n_predictions': len(self.predictions_history)
        }


class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA model for time series forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), 
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize ARIMA model.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit ARIMA model.
        
        Args:
            X: Time index (not used for ARIMA)
            y: Time series values
        """
        try:
            if self.seasonal_order:
                self.model = sm.tsa.SARIMAX(
                    y, 
                    order=self.order, 
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = sm.tsa.ARIMA(y, order=self.order)
            
            self.model = self.model.fit()
            self.is_fitted = True
            logger.info(f"ARIMA model fitted successfully with order {self.order}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Time index (not used)
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.model.forecast(steps=steps)
            return forecast.values if hasattr(forecast, 'values') else forecast
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_type': 'ARIMA',
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'is_fitted': self.is_fitted
        }
        
        if self.is_fitted and self.model:
            info.update({
                'aic': getattr(self.model, 'aic', None),
                'bic': getattr(self.model, 'bic', None),
                'loglikelihood': getattr(self.model, 'llf', None)
            })
        
        return info


class ProphetModel(BaseTimeSeriesModel):
    """Prophet model for time series forecasting."""
    
    def __init__(self, **kwargs):
        """
        Initialize Prophet model.
        
        Args:
            **kwargs: Prophet parameters
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not available. Install with: pip install prophet")
        
        self.model = Prophet(**kwargs)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit Prophet model.
        
        Args:
            X: Time index
            y: Time series values
        """
        # Create DataFrame for Prophet
        df = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
            'y': y
        })
        
        self.model.fit(df)
        self.is_fitted = True
        logger.info("Prophet model fitted successfully")
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Time index
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future DataFrame
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        # Return only the forecasted values
        return forecast['yhat'].tail(steps).values
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'Prophet',
            'is_fitted': self.is_fitted,
            'available': PROPHET_AVAILABLE
        }


class LSTMModel(BaseTimeSeriesModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, sequence_length: int = 20, hidden_size: int = 50, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 epochs: int = 100, batch_size: int = 32):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: Hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit LSTM model.
        
        Args:
            X: Time index (not used)
            y: Time series values
        """
        # Normalize data
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(y_scaled)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_seq).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y_seq)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Define model
        self.model = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Add output layer
        self.output_layer = nn.Linear(self.hidden_size, 1)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.model.parameters()) + list(self.output_layer.parameters()))
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                lstm_out, _ = self.model(batch_X)
                output = self.output_layer(lstm_out[:, -1, :])
                
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.is_fitted = True
        logger.info("LSTM model fitted successfully")
    
    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Time index (not used)
            steps: Number of steps ahead to predict
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        predictions = []
        
        # Use the last sequence_length points for prediction
        last_sequence = X[-self.sequence_length:]
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1)).flatten()
        
        current_input = torch.FloatTensor(last_sequence_scaled).unsqueeze(0).unsqueeze(-1)
        
        with torch.no_grad():
            for _ in range(steps):
                lstm_out, _ = self.model(current_input)
                output = self.output_layer(lstm_out[:, -1, :])
                pred = output.item()
                predictions.append(pred)
                
                # Update input for next prediction
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    output.unsqueeze(0).unsqueeze(0)
                ], dim=1)
        
        # Inverse transform predictions
        predictions_scaled = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions_scaled.flatten()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': 'LSTM',
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'is_fitted': self.is_fitted
        }


class AnomalyDetector:
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1, **kwargs):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            **kwargs: Additional parameters for IsolationForest
        """
        self.model = IsolationForest(contamination=contamination, **kwargs)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit anomaly detector.
        
        Args:
            X: Feature matrix
        """
        self.model.fit(X)
        self.is_fitted = True
        logger.info("Anomaly detector fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores (-1 for anomalies, 1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores.
        
        Args:
            X: Feature matrix
            
        Returns:
            Anomaly scores (lower values indicate more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.score_samples(X)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    y = 2 * t + 1 + np.random.randn(100)
    
    # Test online learning model
    online_model = OnlineLearningModel(window_size=10)
    predictions = online_model.online_predict(y)
    
    print(f"Online learning predictions: {len(predictions)} points")
    print(f"Model info: {online_model.get_model_info()}")
    
    # Test ARIMA model
    arima_model = ARIMAModel(order=(1, 1, 1))
    arima_model.fit(t, y)
    arima_pred = arima_model.predict(t, steps=5)
    
    print(f"ARIMA predictions: {arima_pred}")
    print(f"Model info: {arima_model.get_model_info()}")
