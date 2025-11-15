# Online Learning for Time Series

A comprehensive time series analysis framework featuring online learning, multiple forecasting models, anomaly detection, and interactive visualizations.

## Features

- **Multiple Forecasting Models**: Online Learning, ARIMA, Prophet, LSTM
- **Anomaly Detection**: Isolation Forest-based anomaly detection
- **Interactive Visualizations**: Matplotlib, Seaborn, and Plotly plots
- **Web Interface**: Streamlit dashboard for interactive analysis
- **Synthetic Data Generation**: Realistic time series with trends, seasonality, and noise
- **Comprehensive Testing**: Unit tests for all components
- **Modern Python**: Type hints, docstrings, PEP8 compliance

## Project Structure

```
├── src/                    # Source code modules
│   ├── data_generator.py   # Time series data generation
│   ├── models.py           # Forecasting models
│   ├── visualization.py   # Plotting and visualization
│   └── main.py            # Main application logic
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── tests/                 # Unit tests
│   └── test_time_series.py # Test suite
├── notebooks/             # Jupyter notebooks (optional)
├── data/                  # Data storage
├── models/                # Saved models
├── plots/                 # Generated plots
├── logs/                  # Log files
├── results/               # Analysis results
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
├── streamlit_app.py      # Streamlit web interface
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Online-Learning-for-Time-Series.git
cd Online-Learning-for-Time-Series
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Command Line Interface

Run the complete analysis pipeline:

```bash
python src/main.py
```

This will:
- Generate synthetic time series data
- Train multiple forecasting models
- Detect anomalies
- Create visualizations
- Save results to the `results/` directory

### Web Interface

Launch the interactive Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501` to access the dashboard.

### Programmatic Usage

```python
from src.main import TimeSeriesAnalysisApp

# Initialize the application
app = TimeSeriesAnalysisApp("config/config.yaml")

# Run complete analysis
results = app.run_complete_analysis(
    trend_type="linear",
    seasonality_type="single",
    add_anomalies=True,
    models_to_train=['online_learning', 'arima', 'prophet']
)

# Access results
data = results['data']['dataframe']
model_results = results['models']
anomalies = results['anomalies']
```

## Configuration

The project uses YAML configuration files. Key parameters:

### Data Generation (`config/config.yaml`)

```yaml
data:
  n_points: 1000              # Number of data points
  trend_strength: 2.0         # Strength of trend component
  noise_level: 1.0           # Noise level
  seasonality_period: 50      # Seasonality period
  seasonality_amplitude: 3.0  # Seasonality amplitude
  random_seed: 42             # Random seed for reproducibility
```

### Model Parameters

```yaml
models:
  online_learning:
    window_size: 10          # Sliding window size
    learning_rate: 0.01       # Learning rate
    
  arima:
    order: [1, 1, 1]         # ARIMA order (p, d, q)
    seasonal_order: [1, 1, 1, 12]  # Seasonal order
    
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
    daily_seasonality: false
    
  lstm:
    sequence_length: 20       # Input sequence length
    hidden_size: 50           # Hidden layer size
    num_layers: 2             # Number of LSTM layers
    dropout: 0.2              # Dropout rate
    epochs: 100               # Training epochs
    batch_size: 32            # Batch size
```

## Models

### Online Learning Model

- **Type**: Stochastic Gradient Descent (SGD) Regressor
- **Use Case**: Real-time prediction and adaptation
- **Advantages**: Memory efficient, adapts to new data
- **Parameters**: Window size, learning rate

### ARIMA Model

- **Type**: AutoRegressive Integrated Moving Average
- **Use Case**: Traditional statistical forecasting
- **Advantages**: Interpretable, handles trends and seasonality
- **Parameters**: Order (p, d, q), seasonal order

### Prophet Model

- **Type**: Facebook's forecasting tool
- **Use Case**: Business forecasting with holidays and events
- **Advantages**: Handles missing data, automatic seasonality detection
- **Parameters**: Seasonality settings, holiday effects

### LSTM Model

- **Type**: Long Short-Term Memory neural network
- **Use Case**: Complex pattern recognition in time series
- **Advantages**: Captures long-term dependencies
- **Parameters**: Sequence length, hidden size, layers

### Anomaly Detection

- **Type**: Isolation Forest
- **Use Case**: Detecting unusual patterns in time series
- **Advantages**: Unsupervised, handles high-dimensional data
- **Parameters**: Contamination rate, number of estimators

## Visualization

The project provides comprehensive visualization capabilities:

### Static Plots (Matplotlib/Seaborn)

- Time series with component decomposition
- Forecasting results with confidence intervals
- Anomaly detection visualization
- Model performance comparison
- Distribution plots
- Correlation matrices

### Interactive Plots (Plotly)

- Interactive time series exploration
- Zoomable and pannable plots
- Hover information
- Model comparison dashboards

### Web Dashboard (Streamlit)

- Real-time parameter adjustment
- Model selection and comparison
- Interactive visualizations
- Performance metrics display

## Testing

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Or run individual test modules:

```bash
python tests/test_time_series.py
```

Test coverage includes:
- Data generation functionality
- Model training and prediction
- Anomaly detection
- Visualization components
- Integration tests

## Examples

### Basic Time Series Generation

```python
from src.data_generator import TimeSeriesGenerator

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
```

### Model Training and Prediction

```python
from src.models import OnlineLearningModel, ARIMAModel

# Online Learning
online_model = OnlineLearningModel(window_size=10)
predictions = online_model.online_predict(values)

# ARIMA
arima_model = ARIMAModel(order=(1, 1, 1))
arima_model.fit(t, values)
forecast = arima_model.predict(t, steps=10)
```

### Anomaly Detection

```python
from src.models import AnomalyDetector

detector = AnomalyDetector(contamination=0.1)
features = df[['value', 'trend', 'seasonality', 'noise']].values
detector.fit(features)
anomaly_scores = detector.score_samples(features)
```

### Visualization

```python
from src.visualization import TimeSeriesVisualizer

visualizer = TimeSeriesVisualizer()
fig = visualizer.plot_time_series(df, "My Time Series")
fig = visualizer.plot_forecast(df, predictions, model_name="ARIMA")
```

## Performance Metrics

The framework evaluates models using multiple metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error

## Logging

The application uses Python's logging module with configurable levels:

- **INFO**: General information about the analysis process
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors that may affect results

Logs are saved to `logs/time_series_analysis.log` and displayed in the console.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/ -v`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## Dependencies

### Core Libraries

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing

### Time Series Libraries

- **statsmodels**: Statistical models
- **pmdarima**: Auto-ARIMA
- **prophet**: Facebook's forecasting tool
- **tslearn**: Time series machine learning
- **darts**: Time series forecasting
- **sktime**: Scikit-learn compatible time series

### Deep Learning

- **torch**: PyTorch for neural networks
- **tensorflow**: TensorFlow (optional)

### Visualization

- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plotting
- **streamlit**: Web interface

### Utilities

- **pyyaml**: YAML configuration
- **python-dotenv**: Environment variables
- **tqdm**: Progress bars
- **pytest**: Testing framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Facebook Prophet team for the forecasting library
- Scikit-learn team for machine learning tools
- Streamlit team for the web interface framework
- The Python data science community for excellent libraries

## Support

For questions, issues, or contributions:

1. Check the documentation in this README
2. Review the test cases for usage examples
3. Open an issue on GitHub
4. Contact the maintainers

## Changelog

### Version 1.0.0

- Initial release
- Online learning implementation
- ARIMA, Prophet, LSTM models
- Anomaly detection
- Interactive visualizations
- Streamlit web interface
- Comprehensive testing suite
- Documentation and examples
# Online-Learning-for-Time-Series
