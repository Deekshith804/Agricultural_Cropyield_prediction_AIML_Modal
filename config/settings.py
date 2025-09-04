"""
Configuration settings for the Sustainable Agriculture AI/ML System.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'threaded': True
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 8050,
    'debug': True
}

# ML Model Configuration
ML_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    },
    'test_size': 0.2,
    'cross_validation_folds': 5
}

# Feature Configuration
FEATURE_CONFIG = {
    'required_features': [
        'temperature',
        'rainfall', 
        'humidity',
        'soil_ph',
        'nitrogen',
        'phosphorus',
        'potassium',
        'organic_matter',
        'irrigation_frequency',
        'fertilizer_usage'
    ],
    'target_feature': 'crop_yield',
    'categorical_features': ['crop_type']
}

# Data Validation Ranges
VALIDATION_RANGES = {
    'temperature': {'min': 10, 'max': 40, 'unit': '°C'},
    'rainfall': {'min': 0, 'max': 300, 'unit': 'mm/month'},
    'humidity': {'min': 20, 'max': 100, 'unit': '%'},
    'soil_ph': {'min': 4.0, 'max': 9.0, 'unit': 'pH'},
    'nitrogen': {'min': 0, 'max': 100, 'unit': 'mg/kg'},
    'phosphorus': {'min': 0, 'max': 100, 'unit': 'mg/kg'},
    'potassium': {'min': 0, 'max': 500, 'unit': 'mg/kg'},
    'organic_matter': {'min': 0, 'max': 10, 'unit': '%'},
    'irrigation_frequency': {'min': 0, 'max': 10, 'unit': 'times/week'},
    'fertilizer_usage': {'min': 0, 'max': 500, 'unit': 'kg/hectare'}
}

# Optimal Agricultural Ranges
OPTIMAL_RANGES = {
    'temperature': {'min': 15, 'max': 35, 'optimal': 25, 'unit': '°C'},
    'rainfall': {'min': 50, 'max': 150, 'optimal': 100, 'unit': 'mm/month'},
    'humidity': {'min': 40, 'max': 80, 'optimal': 60, 'unit': '%'},
    'soil_ph': {'min': 6.0, 'max': 7.5, 'optimal': 6.5, 'unit': 'pH'},
    'nitrogen': {'min': 20, 'max': 40, 'optimal': 30, 'unit': 'mg/kg'},
    'phosphorus': {'min': 15, 'max': 25, 'optimal': 20, 'unit': 'mg/kg'},
    'potassium': {'min': 150, 'max': 250, 'optimal': 200, 'unit': 'mg/kg'},
    'organic_matter': {'min': 2.0, 'max': 5.0, 'optimal': 3.5, 'unit': '%'},
    'irrigation_frequency': {'min': 2, 'max': 5, 'optimal': 3, 'unit': 'times/week'},
    'fertilizer_usage': {'min': 50, 'max': 200, 'optimal': 100, 'unit': 'kg/hectare'}
}

# Crop-Specific Requirements
CROP_REQUIREMENTS = {
    'wheat': {
        'optimal_temp': 20,
        'optimal_ph': 6.5,
        'water_needs': 'moderate',
        'nutrient_needs': 'high',
        'growing_season': 'fall-spring',
        'yield_range': {'min': 3.0, 'max': 8.0, 'unit': 'tons/hectare'}
    },
    'corn': {
        'optimal_temp': 25,
        'optimal_ph': 6.0,
        'water_needs': 'high',
        'nutrient_needs': 'very_high',
        'growing_season': 'spring-summer',
        'yield_range': {'min': 8.0, 'max': 15.0, 'unit': 'tons/hectare'}
    },
    'soybeans': {
        'optimal_temp': 22,
        'optimal_ph': 6.8,
        'water_needs': 'moderate',
        'nutrient_needs': 'medium',
        'growing_season': 'spring-summer',
        'yield_range': {'min': 2.5, 'max': 4.5, 'unit': 'tons/hectare'}
    },
    'rice': {
        'optimal_temp': 28,
        'optimal_ph': 6.0,
        'water_needs': 'very_high',
        'nutrient_needs': 'high',
        'growing_season': 'spring-summer',
        'yield_range': {'min': 6.0, 'max': 12.0, 'unit': 'tons/hectare'}
    }
}

# Sustainable Practice Categories
SUSTAINABLE_PRACTICES = {
    'water_conservation': {
        'description': 'Practices to optimize water usage and reduce waste',
        'practices': [
            'Implement drip irrigation systems',
            'Use mulch to reduce evaporation',
            'Schedule irrigation during early morning or evening',
            'Install soil moisture sensors',
            'Practice rainwater harvesting',
            'Use drought-resistant crop varieties',
            'Implement contour farming'
        ]
    },
    'soil_health': {
        'description': 'Practices to improve and maintain soil quality',
        'practices': [
            'Use cover crops to prevent erosion',
            'Implement crop rotation',
            'Add organic matter to soil',
            'Practice no-till farming',
            'Use green manure crops',
            'Maintain proper soil pH',
            'Avoid over-tilling'
        ]
    },
    'nutrient_management': {
        'description': 'Efficient use of fertilizers and soil nutrients',
        'practices': [
            'Conduct regular soil testing',
            'Use slow-release fertilizers',
            'Implement precision agriculture',
            'Use biofertilizers',
            'Practice integrated nutrient management',
            'Apply fertilizers at optimal times',
            'Use crop-specific nutrient requirements'
        ]
    },
    'pest_management': {
        'description': 'Integrated approaches to pest control',
        'practices': [
            'Use integrated pest management (IPM)',
            'Plant pest-resistant crop varieties',
            'Use beneficial insects for pest control',
            'Practice crop rotation',
            'Use organic pesticides when necessary',
            'Monitor pest populations regularly',
            'Implement biological controls'
        ]
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'agriculture_system.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# Database Configuration (for future use)
DATABASE_CONFIG = {
    'type': 'sqlite',  # sqlite, postgresql, mysql
    'host': 'localhost',
    'port': 5432,
    'database': 'agriculture_db',
    'username': 'user',
    'password': 'password'
}

# External API Configuration (for future integrations)
EXTERNAL_APIS = {
    'weather_api': {
        'provider': 'openweathermap',  # openweathermap, accuweather, etc.
        'api_key': os.getenv('WEATHER_API_KEY', ''),
        'base_url': 'https://api.openweathermap.org/data/2.5/',
        'units': 'metric'
    },
    'soil_api': {
        'provider': 'soilgrids',  # soilgrids, usda, etc.
        'api_key': os.getenv('SOIL_API_KEY', ''),
        'base_url': 'https://rest.isric.org/soilgrids/v2.0/'
    }
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_r2_score': 0.7,
    'max_rmse': 1.0,
    'min_cross_validation_score': 0.6,
    'max_feature_importance_variance': 0.3
}

# Sustainability Scoring Weights
SUSTAINABILITY_WEIGHTS = {
    'soil_health': 0.3,
    'water_efficiency': 0.25,
    'nutrient_efficiency': 0.2,
    'pest_management': 0.15,
    'resource_optimization': 0.1
}

# System Monitoring
MONITORING_CONFIG = {
    'health_check_interval': 300,  # seconds
    'model_performance_check_interval': 3600,  # seconds
    'data_quality_check_interval': 86400,  # seconds (daily)
    'max_response_time': 5.0,  # seconds
    'max_memory_usage': 0.8,  # 80% of available memory
    'max_cpu_usage': 0.9  # 90% of available CPU
}
