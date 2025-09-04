#!/usr/bin/env python3
"""
Example script showing how to use the crop yield prediction model directly.
"""

import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_models.crop_yield_predictor import CropYieldPredictor
import pandas as pd

def main():
    """Example of using the crop yield prediction model."""
    print("🌱 Sustainable Agriculture AI/ML System - Yield Prediction Example")
    print("=" * 60)
    
    # Create sample data for prediction
    sample_data = pd.DataFrame([{
        'temperature': 25.0,
        'rainfall': 80.0,
        'humidity': 65.0,
        'soil_ph': 6.5,
        'nitrogen': 30.0,
        'phosphorus': 20.0,
        'potassium': 200.0,
        'organic_matter': 3.0,
        'irrigation_frequency': 3,
        'fertilizer_usage': 100.0
    }])
    
    print("Sample input data:")
    for col in sample_data.columns:
        print(f"  {col}: {sample_data[col].iloc[0]}")
    
    # Try to load a trained model
    try:
        predictor = CropYieldPredictor()
        
        # Try to load an existing model
        model_path = os.path.join('models', 'random_forest_model.pkl')
        if os.path.exists(model_path):
            print(f"\nLoading trained model from {model_path}...")
            predictor.load_model(model_path)
        else:
            print(f"\n❌ No trained model found at {model_path}")
            print("Please train a model first using:")
            print("  python ml_models/train_models.py")
            return
        
        # Make prediction
        print("\nMaking prediction...")
        prediction = predictor.predict(sample_data)
        
        print(f"\n🌾 Predicted Crop Yield: {prediction[0]:.2f} tons/hectare")
        
        # Show feature importance
        print("\n📊 Feature Importance:")
        importance = predictor.get_feature_importance()
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. You have trained the models first")
        print("2. The 'models/' directory contains the trained model files")

if __name__ == "__main__":
    main()