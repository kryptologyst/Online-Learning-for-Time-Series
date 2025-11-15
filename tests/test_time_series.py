"""
Unit tests for time series analysis modules.

This module provides comprehensive tests for all components
of the time series analysis system.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import modules to test
from src.data_generator import TimeSeriesGenerator, load_config
from src.models import (
    OnlineLearningModel, ARIMAModel, ProphetModel, LSTMModel, 
    AnomalyDetector, evaluate_model
)
from src.visualization import TimeSeriesVisualizer, create_dashboard_data


class TestTimeSeriesGenerator(unittest.TestCase):
    """Test cases for TimeSeriesGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'n_points': 100,
            'trend_strength': 2.0,
            'noise_level': 1.0,
            'seasonality_period': 20,
            'seasonality_amplitude': 3.0,
            'random_seed': 42
        }
        self.generator = TimeSeriesGenerator(self.config)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.n_points, 100)
        self.assertEqual(self.generator.trend_strength, 2.0)
        self.assertEqual(self.generator.noise_level, 1.0)
    
    def test_generate_linear_trend(self):
        """Test linear trend generation."""
        t = np.linspace(0, 10, 100)
        trend = self.generator.generate_linear_trend(t)
        
        self.assertEqual(len(trend), 100)
        self.assertAlmostEqual(trend[0], 0.0, places=5)
        self.assertAlmostEqual(trend[-1], 20.0, places=5)
    
    def test_generate_seasonality(self):
        """Test seasonality generation."""
        t = np.linspace(0, 10, 100)
        seasonality = self.generator.generate_seasonality(t)
        
        self.assertEqual(len(seasonality), 100)
        self.assertGreaterEqual(np.min(seasonality), -3.0)
        self.assertLessEqual(np.max(seasonality), 3.0)
    
    def test_generate_noise(self):
        """Test noise generation."""
        noise = self.generator.generate_noise(100)
        
        self.assertEqual(len(noise), 100)
        self.assertAlmostEqual(np.mean(noise), 0.0, places=1)
        self.assertAlmostEqual(np.std(noise), 1.0, places=1)
    
    def test_generate_time_series(self):
        """Test complete time series generation."""
        t, values, df = self.generator.generate_time_series("linear", "single")
        
        self.assertEqual(len(t), 100)
        self.assertEqual(len(values), 100)
        self.assertEqual(len(df), 100)
        self.assertIn('time', df.columns)
        self.assertIn('value', df.columns)
        self.assertIn('trend', df.columns)
        self.assertIn('seasonality', df.columns)
        self.assertIn('noise', df.columns)
    
    def test_generate_time_series_with_anomalies(self):
        """Test time series generation with anomalies."""
        t, values, df = self.generator.generate_time_series("linear", "single", add_anomalies=True)
        
        self.assertIn('is_anomaly', df.columns)
        self.assertTrue(df['is_anomaly'].dtype == bool)
    
    def test_generate_multiple_series(self):
        """Test multiple series generation."""
        series_dict = self.generator.generate_multiple_series()
        
        self.assertIn("linear_single", series_dict)
        self.assertIn("complex_multiple", series_dict)
        self.assertIn("linear_with_anomalies", series_dict)
        
        for series_name, df in series_dict.items():
            self.assertEqual(len(df), 100)
            self.assertIn('value', df.columns)


class TestOnlineLearningModel(unittest.TestCase):
    """Test cases for OnlineLearningModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = OnlineLearningModel(window_size=5, learning_rate=0.01)
        self.time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.window_size, 5)
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertFalse(self.model.is_fitted)
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([1, 2, 3, 4, 5])
        
        self.model.fit(X, y)
        self.assertTrue(self.model.is_fitted)
        
        predictions = self.model.predict(np.array([[6]]))
        self.assertEqual(len(predictions), 1)
    
    def test_online_predict(self):
        """Test online prediction."""
        predictions = self.model.online_predict(self.time_series)
        
        self.assertEqual(len(predictions), 5)  # Should have 5 predictions
        self.assertTrue(self.model.is_fitted)
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.model.get_model_info()
        
        self.assertIn('model_type', info)
        self.assertIn('window_size', info)
        self.assertIn('learning_rate', info)
        self.assertIn('is_fitted', info)


class TestARIMAModel(unittest.TestCase):
    """Test cases for ARIMAModel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = ARIMAModel(order=(1, 1, 1))
        # Create simple time series data
        np.random.seed(42)
        self.time_series = np.cumsum(np.random.randn(50)) + np.linspace(0, 10, 50)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.order, (1, 1, 1))
        self.assertFalse(self.model.is_fitted)
    
    def test_fit_and_predict(self):
        """Test model fitting and prediction."""
        t = np.arange(len(self.time_series))
        
        self.model.fit(t, self.time_series)
        self.assertTrue(self.model.is_fitted)
        
        predictions = self.model.predict(t, steps=5)
        self.assertEqual(len(predictions), 5)
    
    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.model.get_model_info()
        
        self.assertIn('model_type', info)
        self.assertIn('order', info)
        self.assertIn('is_fitted', info)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector(contamination=0.1)
        # Create test data with some anomalies
        np.random.seed(42)
        normal_data = np.random.randn(90, 4)
        anomaly_data = np.random.randn(10, 4) + 5  # Shifted anomalies
        self.features = np.vstack([normal_data, anomaly_data])
    
    def test_initialization(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.model.contamination, 0.1)
        self.assertFalse(self.detector.is_fitted)
    
    def test_fit_and_predict(self):
        """Test detector fitting and prediction."""
        self.detector.fit(self.features)
        self.assertTrue(self.detector.is_fitted)
        
        predictions = self.detector.predict(self.features)
        self.assertEqual(len(predictions), len(self.features))
        
        # Should have some anomalies detected
        anomaly_count = np.sum(predictions == -1)
        self.assertGreater(anomaly_count, 0)
    
    def test_score_samples(self):
        """Test anomaly scoring."""
        self.detector.fit(self.features)
        
        scores = self.detector.score_samples(self.features)
        self.assertEqual(len(scores), len(self.features))
        
        # Lower scores should indicate more anomalous
        self.assertTrue(np.all(scores <= 0))


class TestTimeSeriesVisualizer(unittest.TestCase):
    """Test cases for TimeSeriesVisualizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = TimeSeriesVisualizer()
        
        # Create test data
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        trend = 2 * t
        seasonality = 3 * np.sin(2 * np.pi * t / 5)
        noise = np.random.randn(100)
        
        self.df = pd.DataFrame({
            'time': t,
            'value': trend + seasonality + noise,
            'trend': trend,
            'seasonality': seasonality,
            'noise': noise
        })
    
    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertEqual(self.visualizer.figsize, (12, 8))
        self.assertEqual(self.visualizer.dpi, 100)
        self.assertIn('primary', self.visualizer.colors)
    
    def test_plot_time_series(self):
        """Test time series plotting."""
        fig = self.visualizer.plot_time_series(self.df, "Test Plot")
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)  # Should have 2 subplots
    
    def test_plot_forecast(self):
        """Test forecast plotting."""
        predictions = np.array([20, 21, 22, 23, 24])
        fig = self.visualizer.plot_forecast(self.df, predictions, model_name="Test Model")
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
    
    def test_plot_anomalies(self):
        """Test anomaly plotting."""
        anomaly_scores = np.random.randn(100)
        fig = self.visualizer.plot_anomalies(self.df, anomaly_scores)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 2)
    
    def test_plot_distribution(self):
        """Test distribution plotting."""
        fig = self.visualizer.plot_distribution(self.df)
        
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 4)  # Should have 4 subplots for 4 components


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def test_evaluate_model(self):
        """Test model evaluation metrics."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        metrics = evaluate_model(y_true, y_pred)
        
        self.assertIn('MSE', metrics)
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertIn('MAPE', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['R2'], 0.9)  # Should be high for good predictions
        self.assertLess(metrics['RMSE'], 1.0)   # Should be low for good predictions


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_dashboard_data(self):
        """Test dashboard data creation."""
        model_results = {
            'model1': {'metrics': {'RMSE': 1.0, 'MAE': 0.8, 'R2': 0.9, 'MAPE': 5.0}},
            'model2': {'metrics': {'RMSE': 1.2, 'MAE': 1.0, 'R2': 0.85, 'MAPE': 6.0}}
        }
        
        dashboard_data = create_dashboard_data(model_results)
        
        self.assertIsInstance(dashboard_data, pd.DataFrame)
        self.assertEqual(len(dashboard_data), 2)
        self.assertIn('Model', dashboard_data.columns)
        self.assertIn('RMSE', dashboard_data.columns)
        self.assertIn('R2', dashboard_data.columns)
    
    def test_load_config(self):
        """Test configuration loading."""
        # Create temporary config file
        config_content = """
data:
  n_points: 100
  trend_strength: 2.0
models:
  online_learning:
    window_size: 10
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_path = f.name
        
        try:
            config = load_config(temp_path)
            self.assertIn('data', config)
            self.assertIn('models', config)
            self.assertEqual(config['data']['n_points'], 100)
        finally:
            os.unlink(temp_path)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        config = {
            'n_points': 50,
            'trend_strength': 1.0,
            'noise_level': 0.5,
            'seasonality_period': 10,
            'seasonality_amplitude': 2.0,
            'random_seed': 42
        }
        
        generator = TimeSeriesGenerator(config)
        t, values, df = generator.generate_time_series("linear", "single")
        
        # Train model
        model = OnlineLearningModel(window_size=5)
        predictions = model.online_predict(values)
        
        # Detect anomalies
        detector = AnomalyDetector()
        features = df[['value', 'trend', 'seasonality', 'noise']].values
        detector.fit(features)
        anomaly_scores = detector.score_samples(features)
        
        # Create visualization
        visualizer = TimeSeriesVisualizer()
        fig = visualizer.plot_time_series(df)
        
        # Verify results
        self.assertEqual(len(values), 50)
        self.assertEqual(len(predictions), 45)  # 50 - 5 (window_size)
        self.assertEqual(len(anomaly_scores), 50)
        self.assertIsNotNone(fig)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestTimeSeriesGenerator,
        TestOnlineLearningModel,
        TestARIMAModel,
        TestAnomalyDetector,
        TestTimeSeriesVisualizer,
        TestEvaluationMetrics,
        TestUtilityFunctions,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
