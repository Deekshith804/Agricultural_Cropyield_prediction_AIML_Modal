# Sustainable Agriculture AI/ML System - Usage Guide

This guide provides detailed instructions on how to use the Sustainable Agriculture AI/ML System, including how to work with real agricultural datasets.

## Table of Contents
1. [System Overview](#system-overview)
2. [Installation](#installation)
3. [Training Models with Real Data](#training-models-with-real-data)
4. [Using Public Agricultural Datasets](#using-public-agricultural-datasets)
5. [API Usage](#api-usage)
6. [Dashboard Usage](#dashboard-usage)
7. [Adding Your Own Dataset](#adding-your-own-dataset)
8. [Troubleshooting](#troubleshooting)

## System Overview

The Sustainable Agriculture AI/ML System is designed to:
- Predict crop yields based on environmental and soil conditions
- Provide sustainable farming practice recommendations
- Analyze soil health and weather patterns
- Optimize resource usage (water, fertilizers, etc.)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Standard Installation
```bash
pip install -r requirements.txt
```

### Windows Installation
If you encounter compilation errors on Windows:
```bash
python install_windows.py
```

## Training Models with Real Data

The system automatically detects and uses real agricultural datasets placed in the `data/` directory. If no datasets are found, it generates sample data for demonstration.

### Quick Start
1. Place your dataset CSV files in the `data/` directory
2. Run the training script:
   ```bash
   python ml_models/train_models.py
   ```

### Training Process
The training script will:
1. Check for real datasets in the `data/` directory
2. Automatically detect dataset types (FAO, Kaggle, India Government, etc.)
3. Preprocess the data to match the required format
4. Train both Random Forest and Gradient Boosting models
5. Save trained models to the `models/` directory

## Using Public Agricultural Datasets

### FAO Crop Yield Dataset
1. Download from the FAO statistical database
2. Place the CSV file in the `data/` directory
3. The system will automatically detect and use it

Example:
```python
# The system will automatically load FAO datasets
# No special configuration needed
```

### Kaggle Crop Production Data
1. Download any crop production dataset from Kaggle
2. Place the CSV file in the `data/` directory
3. The system will automatically detect and use it

### India Government Agriculture Data
1. Obtain datasets from Indian government agricultural departments
2. Place the CSV file in the `data/` directory
3. The system will automatically detect and use it

## API Usage

The system provides a RESTful API for integration with other applications.

### Starting the API Server
```bash
python api/app.py
```

### API Endpoints

#### 1. Health Check
```
GET /health
```
Response:
```json
{
  "status": "healthy",
  "timestamp": "2023-01-01T00:00:00Z"
}
```

#### 2. Crop Yield Prediction
```
POST /predict/yield
```
Request:
```json
{
  "temperature": 25.0,
  "rainfall": 80.0,
  "humidity": 65.0,
  "soil_ph": 6.5,
  "nitrogen": 30.0,
  "phosphorus": 20.0,
  "potassium": 200.0,
  "organic_matter": 3.0,
  "irrigation_frequency": 3,
  "fertilizer_usage": 100.0,
  "crop_type": "wheat"
}
```

Response:
```json
{
  "predicted_yield": 5.2,
  "unit": "tons/hectare",
  "confidence": 0.85,
  "feature_importance": {
    "temperature": 0.15,
    "rainfall": 0.12,
    "humidity": 0.08,
    "soil_ph": 0.10,
    "nitrogen": 0.18,
    "phosphorus": 0.12,
    "potassium": 0.08,
    "organic_matter": 0.10,
    "irrigation_frequency": 0.05,
    "fertilizer_usage": 0.02
  }
}
```

#### 3. Sustainable Practice Recommendations
```
POST /recommend/practices
```
Request:
```json
{
  "soil_conditions": {
    "organic_matter": 2.5,
    "nitrogen": 25.0,
    "phosphorus": 18.0,
    "potassium": 180.0
  },
  "weather_conditions": {
    "rainfall": 50.0,
    "temperature": 28.0,
    "humidity": 45.0
  },
  "crop_type": "wheat"
}
```

Response:
```json
{
  "water_management": {
    "priority": "high",
    "explanation": "Low rainfall detected, recommend water conservation practices",
    "practices": [
      "Install drip irrigation system",
      "Apply mulch to reduce evaporation",
      "Schedule irrigation during cooler hours"
    ]
  },
  "soil_health": {
    "priority": "medium",
    "explanation": "Soil nutrient levels could be improved",
    "practices": [
      "Add organic compost",
      "Practice crop rotation",
      "Reduce tillage to preserve soil structure"
    ]
  },
  "fertilizer_optimization": {
    "priority": "medium",
    "explanation": "Balanced nutrient approach recommended",
    "practices": [
      "Use organic fertilizers instead of chemical ones",
      "Apply fertilizers based on soil test results",
      "Consider slow-release fertilizer options"
    ]
  }
}
```

## Dashboard Usage

The system includes an interactive web-based dashboard for easy access to all features.

### Starting the Dashboard
```bash
python dashboard/app.py
```

### Dashboard Features
1. **Crop Yield Prediction**: Input environmental conditions to predict crop yields
2. **Sustainable Practice Recommendations**: Get AI-powered recommendations for sustainable farming
3. **Data Analysis**: Visualize agricultural data and trends
4. **System Status**: Monitor the health of the ML models and API

### Using the Dashboard
1. Open your browser and navigate to `http://localhost:8050`
2. Use the navigation tabs to access different features:
   - **Crop Yield Prediction**: Enter soil and weather data to predict yields
   - **Sustainable Practices**: Get recommendations based on conditions
   - **Data Analysis**: View charts and graphs of agricultural data
   - **System Status**: Check the status of all system components

## Adding Your Own Dataset

To use your own agricultural dataset:

### Required Columns
Your CSV file must contain these columns:
- `temperature`: Temperature in Celsius
- `rainfall`: Rainfall in mm/month
- `humidity`: Humidity percentage
- `soil_ph`: Soil pH level
- `nitrogen`: Nitrogen content in mg/kg
- `phosphorus`: Phosphorus content in mg/kg
- `potassium`: Potassium content in mg/kg
- `organic_matter`: Organic matter percentage
- `irrigation_frequency`: Irrigation frequency (times/week)
- `fertilizer_usage`: Fertilizer usage in kg/hectare
- `crop_type`: Type of crop (wheat, corn, soybeans, rice)
- `crop_yield`: Target variable (crop yield in tons/hectare)

### Example CSV Format
```csv
temperature,rainfall,humidity,soil_ph,nitrogen,phosphorus,potassium,organic_matter,irrigation_frequency,fertilizer_usage,crop_type,crop_yield
25.0,80.0,65.0,6.5,30.0,20.0,200.0,3.0,3,100.0,wheat,5.2
22.0,100.0,70.0,6.8,35.0,25.0,220.0,3.5,4,120.0,corn,6.1
```

### Steps to Use Your Dataset
1. Place your CSV file in the `data/` directory
2. Run the training script:
   ```bash
   python ml_models/train_models.py
   ```
3. The system will automatically detect and use your dataset

## Troubleshooting

### Common Issues

#### 1. Missing Dependencies
If you get import errors, install dependencies:
```bash
pip install -r requirements.txt
```

#### 2. Model Loading Errors
If models fail to load:
1. Delete the `models/` directory
2. Re-run training:
   ```bash
   python ml_models/train_models.py
   ```

#### 3. API Connection Issues
If the dashboard can't connect to the API:
1. Ensure the API server is running:
   ```bash
   python api/app.py
   ```
2. Check that the API is accessible at `http://localhost:5000`

#### 4. Dataset Format Issues
If you get errors with your dataset:
1. Ensure all required columns are present
2. Check that numeric values are properly formatted
3. Verify there are no missing values

### Getting Help
For additional help, check:
- README.md for basic installation and usage
- This usage guide for detailed instructions
- GitHub issues for known problems and solutions