import pandas as pd
import numpy as np
import os
from crop_yield_predictor import CropYieldPredictor, SustainablePracticeRecommender

def generate_sample_data(n_samples=1000):
    """
    Generate synthetic agricultural data for training the models.
    In a real scenario, this would be replaced with actual historical data.
    """
    np.random.seed(42)
    
    # Generate realistic agricultural data
    data = {
        'temperature': np.random.normal(25, 8, n_samples),  # Celsius
        'rainfall': np.random.exponential(50, n_samples),   # mm/month
        'humidity': np.random.uniform(40, 90, n_samples),   # %
        'soil_ph': np.random.normal(6.5, 1.0, n_samples),  # pH scale
        'nitrogen': np.random.normal(30, 10, n_samples),    # mg/kg
        'phosphorus': np.random.normal(20, 8, n_samples),   # mg/kg
        'potassium': np.random.normal(200, 50, n_samples),  # mg/kg
        'organic_matter': np.random.normal(3.0, 1.0, n_samples),  # %
        'irrigation_frequency': np.random.poisson(3, n_samples),   # times/week
        'fertilizer_usage': np.random.exponential(100, n_samples), # kg/hectare
        'crop_type': np.random.choice(['wheat', 'corn', 'soybeans', 'rice'], n_samples)
    }
    
    # Create target variable (crop yield) based on features
    # This is a simplified model - in reality, the relationship would be more complex
    base_yield = 5.0  # tons/hectare
    
    # Temperature effect (optimal around 25°C)
    temp_effect = -0.1 * (data['temperature'] - 25)**2 + 2.0
    
    # Rainfall effect (optimal around 100mm/month)
    rain_effect = -0.0001 * (data['rainfall'] - 100)**2 + 1.5
    
    # Soil pH effect (optimal around 6.5)
    ph_effect = -0.5 * (data['soil_ph'] - 6.5)**2 + 1.0
    
    # Nutrient effects
    nutrient_effect = (
        0.02 * data['nitrogen'] + 
        0.03 * data['phosphorus'] + 
        0.001 * data['potassium']
    ) / 100
    
    # Organic matter effect
    om_effect = 0.3 * data['organic_matter']
    
    # Irrigation effect
    irrigation_effect = 0.1 * data['irrigation_frequency']
    
    # Fertilizer effect (diminishing returns)
    fertilizer_effect = 0.5 * (1 - np.exp(-0.01 * data['fertilizer_usage']))
    
    # Calculate final yield with some noise
    yield_prediction = (
        base_yield + temp_effect + rain_effect + ph_effect + 
        nutrient_effect + om_effect + irrigation_effect + fertilizer_effect
    )
    
    # Add realistic noise
    noise = np.random.normal(0, 0.5, n_samples)
    data['crop_yield'] = np.maximum(0, yield_prediction + noise)
    
    return pd.DataFrame(data)

def train_crop_yield_model():
    """
    Train the crop yield prediction model.
    """
    # Try to load real agricultural datasets first
    try:
        # Add the utils directory to the path
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
        
        from data_processor import load_sample_dataset
        
        print("Loading agricultural dataset...")
        data = load_sample_dataset()
        
        if data.empty:
            print("No dataset found, generating sample agricultural data...")
            data = generate_sample_data(2000)
            print(f"Generated {len(data)} samples")
        else:
            print(f"Loaded {len(data)} samples from dataset")
    except Exception as e:
        print(f"Error loading dataset, generating sample data: {e}")
        data = generate_sample_data(2000)
        print(f"Generated {len(data)} samples")
    
    print(f"Data shape: {data.shape}")
    print("\nSample data:")
    print(data.head())
    
    # Prepare features and target
    feature_columns = [
        'temperature', 'rainfall', 'humidity', 'soil_ph',
        'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
        'irrigation_frequency', 'fertilizer_usage'
    ]
    
    X = data[feature_columns]
    y = data['crop_yield']
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_predictor = CropYieldPredictor(model_type='random_forest')
    rf_metrics = rf_predictor.train(X, y)
    
    print("Random Forest Model Performance:")
    for metric, value in rf_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Train Gradient Boosting model
    print("\nTraining Gradient Boosting model...")
    gb_predictor = CropYieldPredictor(model_type='gradient_boosting')
    gb_metrics = gb_predictor.train(X, y)
    
    print("Gradient Boosting Model Performance:")
    for metric, value in gb_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Get feature importance
    print("\nFeature Importance (Random Forest):")
    feature_importance = rf_predictor.get_feature_importance()
    for feature, importance in feature_importance.items():
        print(f"  {feature}: {importance:.4f}")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    rf_predictor.save_model('models/random_forest_model.pkl')
    gb_predictor.save_model('models/gradient_boosting_model.pkl')
    
    print("\nModels saved successfully!")
    
    # Test predictions
    print("\nTesting predictions with sample data...")
    test_data = data.head(5)
    predictions = rf_predictor.predict(test_data)
    
    print("\nSample Predictions:")
    for i, (actual, predicted) in enumerate(zip(test_data['crop_yield'], predictions)):
        print(f"  Sample {i+1}: Actual: {actual:.2f}, Predicted: {predicted:.2f}")
    
    return rf_predictor, gb_predictor

def test_sustainable_practices():
    """
    Test the sustainable practices recommender.
    """
    print("\n" + "="*50)
    print("Testing Sustainable Practices Recommender")
    print("="*50)
    
    recommender = SustainablePracticeRecommender()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Low Rainfall Scenario',
            'soil': {'organic_matter': 1.5, 'nitrogen': 25, 'phosphorus': 18, 'potassium': 180},
            'weather': {'rainfall': 30, 'temperature': 28, 'humidity': 45},
            'crop': 'wheat'
        },
        {
            'name': 'Poor Soil Health Scenario',
            'soil': {'organic_matter': 1.0, 'nitrogen': 15, 'phosphorus': 12, 'potassium': 120},
            'weather': {'rainfall': 80, 'temperature': 22, 'humidity': 70},
            'crop': 'corn'
        },
        {
            'name': 'Optimal Conditions Scenario',
            'soil': {'organic_matter': 4.0, 'nitrogen': 35, 'phosphorus': 25, 'potassium': 250},
            'weather': {'rainfall': 100, 'temperature': 24, 'humidity': 65},
            'crop': 'soybeans'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)
        
        recommendations = recommender.recommend_practices(
            scenario['soil'], 
            scenario['weather'], 
            scenario['crop']
        )
        
        for category, rec in recommendations.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Priority: {rec['priority']}")
            print(f"  Explanation: {rec['explanation']}")
            print("  Practices:")
            for practice in rec['practices']:
                print(f"    - {practice}")

if __name__ == "__main__":
    print("Sustainable Agriculture AI/ML System")
    print("=" * 50)
    
    # Train models
    rf_model, gb_model = train_crop_yield_model()
    
    # Test sustainable practices
    test_sustainable_practices()
    
    print("\n" + "="*50)
    print("Training and Testing Complete!")
    print("="*50)
    print("\nNext steps:")
    print("1. Models are saved in 'models/' directory")
    print("2. Use the trained models for predictions")
    print("3. Integrate with the web API and dashboard")
    print("4. Fine-tune models with real agricultural data")
