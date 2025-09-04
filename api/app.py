from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import ML models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.crop_yield_predictor import CropYieldPredictor, SustainablePracticeRecommender

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
crop_predictor = None
practice_recommender = None

def load_models():
    """Load the trained ML models."""
    global crop_predictor, practice_recommender
    
    try:
        # Load Random Forest model (you can change this to load other models)
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_model.pkl')
        if os.path.exists(model_path):
            crop_predictor = CropYieldPredictor()
            crop_predictor.load_model(model_path)
            print("ML models loaded successfully!")
        else:
            print("Warning: No trained models found. Please run train_models.py first.")
            crop_predictor = None
            
        practice_recommender = SustainablePracticeRecommender()
        
    except Exception as e:
        print(f"Error loading models: {e}")
        crop_predictor = None
        practice_recommender = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': crop_predictor is not None,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/predict/yield', methods=['POST'])
def predict_crop_yield():
    """
    Predict crop yield based on environmental and soil conditions.
    
    Expected JSON payload:
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
        "fertilizer_usage": 100.0
    }
    """
    if crop_predictor is None:
        return jsonify({
            'error': 'ML models not loaded. Please ensure models are trained first.'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = [
            'temperature', 'rainfall', 'humidity', 'soil_ph',
            'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
            'irrigation_frequency', 'fertilizer_usage'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Create DataFrame for prediction
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = crop_predictor.predict(input_data)[0]
        
        # Get feature importance
        feature_importance = crop_predictor.get_feature_importance()
        
        # Calculate confidence score (simplified - in reality, this would be more sophisticated)
        confidence = 0.85  # Placeholder
        
        return jsonify({
            'predicted_yield': round(prediction, 2),
            'unit': 'tons/hectare',
            'confidence': confidence,
            'feature_importance': feature_importance,
            'input_parameters': data,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/recommend/practices', methods=['POST'])
def recommend_sustainable_practices():
    """
    Recommend sustainable agricultural practices based on current conditions.
    
    Expected JSON payload:
    {
        "soil_conditions": {
            "organic_matter": 2.5,
            "nitrogen": 25.0,
            "phosphorus": 18.0,
            "potassium": 180.0
        },
        "weather_conditions": {
            "rainfall": 60.0,
            "temperature": 28.0,
            "humidity": 45.0
        },
        "crop_type": "wheat"
    }
    """
    if practice_recommender is None:
        return jsonify({
            'error': 'Practice recommender not available'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_sections = ['soil_conditions', 'weather_conditions', 'crop_type']
        missing_sections = [section for section in required_sections if section not in data]
        
        if missing_sections:
            return jsonify({
                'error': f'Missing required sections: {missing_sections}'
            }), 400
        
        # Get recommendations
        recommendations = practice_recommender.recommend_practices(
            data['soil_conditions'],
            data['weather_conditions'],
            data['crop_type']
        )
        
        return jsonify({
            'recommendations': recommendations,
            'crop_type': data['crop_type'],
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Recommendation failed: {str(e)}'
        }), 500

@app.route('/soil/analysis', methods=['GET'])
def get_soil_analysis():
    """
    Get general soil health insights and recommendations.
    """
    try:
        soil_insights = {
            'optimal_ranges': {
                'soil_ph': {'min': 6.0, 'max': 7.5, 'optimal': 6.5},
                'nitrogen': {'min': 20, 'max': 40, 'optimal': 30, 'unit': 'mg/kg'},
                'phosphorus': {'min': 15, 'max': 25, 'optimal': 20, 'unit': 'mg/kg'},
                'potassium': {'min': 150, 'max': 250, 'optimal': 200, 'unit': 'mg/kg'},
                'organic_matter': {'min': 2.0, 'max': 5.0, 'optimal': 3.5, 'unit': '%'}
            },
            'soil_health_tips': [
                'Maintain soil pH between 6.0-7.5 for most crops',
                'Regular soil testing helps identify nutrient deficiencies',
                'Organic matter improves soil structure and water retention',
                'Crop rotation helps maintain soil fertility',
                'Cover crops prevent soil erosion and add organic matter'
            ],
            'testing_frequency': 'Test soil every 2-3 years or when changing crops',
            'sampling_depth': 'Sample from 0-6 inches for most crops'
        }
        
        return jsonify({
            'soil_insights': soil_insights,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Soil analysis failed: {str(e)}'
        }), 500

@app.route('/weather/forecast', methods=['GET'])
def get_weather_forecast():
    """
    Get weather-based agricultural insights.
    Note: This is a placeholder - in a real system, this would integrate with weather APIs.
    """
    try:
        # Placeholder weather insights
        weather_insights = {
            'current_season': 'Spring',
            'general_recommendations': [
                'Monitor soil moisture levels regularly',
                'Adjust irrigation based on rainfall forecasts',
                'Protect crops from late frost if applicable',
                'Plan planting based on soil temperature',
                'Consider wind protection for young plants'
            ],
            'seasonal_tips': {
                'spring': 'Focus on soil preparation and early planting',
                'summer': 'Monitor water needs and pest pressure',
                'fall': 'Harvest and prepare for winter crops',
                'winter': 'Plan for next season and maintain equipment'
            }
        }
        
        return jsonify({
            'weather_insights': weather_insights,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Weather forecast failed: {str(e)}'
        }), 500

@app.route('/models/status', methods=['GET'])
def get_model_status():
    """Get the status of loaded ML models."""
    return jsonify({
        'crop_yield_predictor': {
            'loaded': crop_predictor is not None,
            'type': crop_predictor.model_type if crop_predictor else None,
            'trained': crop_predictor.is_trained if crop_predictor else False
        },
        'practice_recommender': {
            'loaded': practice_recommender is not None
        },
        'timestamp': pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    print("Loading ML models...")
    load_models()
    
    print("Starting Sustainable Agriculture API...")
    app.run(debug=True, host='0.0.0.0', port=5000)
