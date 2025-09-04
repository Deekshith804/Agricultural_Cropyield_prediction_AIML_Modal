# Sustainable Agriculture AI/ML System

An intelligent system that predicts crop yields and suggests sustainable agricultural practices using machine learning and historical data analysis.

## Features

- **Crop Yield Prediction**: ML models that forecast crop production based on environmental conditions
- **Sustainable Practice Recommendations**: AI-powered suggestions for resource optimization
- **Soil Health Analysis**: Insights on soil nutrient management and improvement
- **Water Usage Optimization**: Recommendations for efficient irrigation practices
- **Interactive Dashboard**: Web-based interface for farmers and agricultural experts
- **Historical Data Analysis**: Trend analysis and pattern recognition

## System Architecture

```
├── ml_models/           # Machine learning models and training scripts
├── data/               # Sample datasets and data processing
├── api/                # Flask API endpoints
├── dashboard/          # Interactive web dashboard
├── utils/              # Utility functions and helpers
└── notebooks/          # Jupyter notebooks for exploration
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Windows Installation (Recommended)

If you're on Windows and encounter compilation errors, use the Windows installation script:

```bash
python install_windows.py
```

This script provides multiple installation methods:
1. **Pip installation** (recommended)
2. **Conda installation** (if available)
3. **Pre-compiled wheels** (for problematic packages)
4. **Visual C++ Build Tools** (instructions)

### Standard Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter issues, try the minimal requirements:
   ```bash
   pip install -r requirements_minimal.txt
   ```

2. **Alternative: Use Conda** (if available):
   ```bash
   conda create -n sustainable_agriculture python=3.9
   conda activate sustainable_agriculture
   conda install scikit-learn pandas numpy flask dash plotly
   ```

### Troubleshooting Windows Issues

If you get "Microsoft Visual C++ 14.0 or greater is required" errors:

1. **Install Visual C++ Build Tools**:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Or use: `winget install Microsoft.VisualStudio.2022.BuildTools`

2. **Use pre-compiled wheels**:
   ```bash
   pip install --only-binary=all scikit-learn pandas numpy
   ```

3. **Use conda instead of pip**:
   ```bash
   conda install scikit-learn pandas numpy
   ```

## Quick Start

1. **Run the API Server**:
   ```bash
   python api/app.py
   ```

2. **Launch the Dashboard**:
   ```bash
   python dashboard/app.py
   ```

3. **Train Models**:
   ```bash
   python ml_models/train_models.py
   ```

4. **Use the Main System**:
   ```bash
   python run_system.py
   ```

## API Endpoints

- `POST /predict/yield` - Predict crop yield
- `POST /recommend/practices` - Get sustainable practice recommendations
- `GET /soil/analysis` - Soil health insights
- `GET /weather/forecast` - Weather-based predictions

## Models

- **Random Forest** for crop yield prediction
- **Gradient Boosting** for practice recommendations
- **Neural Networks** for complex pattern recognition
- **Time Series Analysis** for seasonal trends

## Data Sources

The system can work with various agricultural datasets including:
- Weather data (temperature, rainfall, humidity)
- Soil composition (pH, nutrients, organic matter)
- Crop management practices
- Historical yield data
- Satellite imagery (optional)

### Supported Public Datasets

1. **FAO Crop Yield Dataset**
   - Download from FAO statistical database
   - Place CSV file in the `data/` directory
   - The system will automatically detect and use it

2. **Kaggle Crop Production Data**
   - Various datasets available on Kaggle
   - Download and place CSV file in the `data/` directory
   - The system will automatically detect and use it

3. **India Government Agriculture Data**
   - Available from Indian government agricultural departments
   - Place CSV file in the `data/` directory
   - The system will automatically detect and use it

### Using Your Own Dataset

To use your own agricultural dataset:
1. Place your CSV file in the `data/` directory
2. Ensure it contains the required columns:
   - `temperature`, `rainfall`, `humidity`, `soil_ph`
   - `nitrogen`, `phosphorus`, `potassium`, `organic_matter`
   - `irrigation_frequency`, `fertilizer_usage`
   - `crop_type` and `crop_yield` (target variable)
3. Run the training script - it will automatically detect and use your dataset

## Documentation and Examples

For detailed usage instructions, see:
- [Usage Guide](docs/usage_guide.md) - Comprehensive documentation with examples

Example scripts are available in the [examples/](examples/) directory:
- [predict_yield.py](examples/predict_yield.py) - Direct model usage for yield prediction
- [recommend_practices.py](examples/recommend_practices.py) - Sustainable practice recommendations
- [load_dataset.py](examples/load_dataset.py) - Loading and preprocessing datasets

## Contributing

This is an open-source project. Contributions are welcome!

## License

MIT License
