import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

class CropYieldPredictor:
    """
    Machine Learning model for predicting crop yields based on environmental and soil conditions.
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, data):
        """
        Prepare features for the model. Expected columns:
        - temperature, rainfall, humidity, soil_ph, nitrogen, phosphorus, potassium
        - organic_matter, irrigation_frequency, fertilizer_usage
        """
        feature_columns = [
            'temperature', 'rainfall', 'humidity', 'soil_ph',
            'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
            'irrigation_frequency', 'fertilizer_usage'
        ]
        
        # Handle missing values - only for numeric columns
        numeric_data = data[feature_columns]
        data[feature_columns] = numeric_data.fillna(numeric_data.mean())
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in data.columns:
                data[col] = 0  # Default value for missing features
        
        return data[feature_columns]
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """
        Train the crop yield prediction model.
        
        Args:
            X: Feature DataFrame
            y: Target variable (crop yield)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        # Prepare features
        X_processed = self.prepare_features(X)
        self.feature_names = X_processed.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def predict(self, X):
        """
        Predict crop yield for new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted crop yields
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features)
    
    def save_model(self, filepath):
        """
        Save the trained model and scaler.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model and scaler.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True

class SustainablePracticeRecommender:
    """
    Recommends sustainable agricultural practices based on current conditions.
    """
    
    def __init__(self):
        self.practices = {
            'water_conservation': [
                'Implement drip irrigation systems',
                'Use mulch to reduce evaporation',
                'Schedule irrigation during early morning or evening',
                'Install soil moisture sensors',
                'Practice rainwater harvesting'
            ],
            'soil_health': [
                'Use cover crops to prevent erosion',
                'Implement crop rotation',
                'Add organic matter to soil',
                'Practice no-till farming',
                'Use green manure crops'
            ],
            'nutrient_management': [
                'Conduct regular soil testing',
                'Use slow-release fertilizers',
                'Implement precision agriculture',
                'Use biofertilizers',
                'Practice integrated nutrient management'
            ],
            'pest_management': [
                'Use integrated pest management (IPM)',
                'Plant pest-resistant crop varieties',
                'Use beneficial insects for pest control',
                'Practice crop rotation',
                'Use organic pesticides when necessary'
            ]
        }
    
    def recommend_practices(self, soil_conditions, weather_conditions, crop_type):
        """
        Recommend sustainable practices based on current conditions.
        
        Args:
            soil_conditions: Dictionary with soil parameters
            weather_conditions: Dictionary with weather parameters
            crop_type: Type of crop being grown
            
        Returns:
            Dictionary with recommended practices and explanations
        """
        recommendations = {}
        
        # Water conservation recommendations
        if weather_conditions.get('rainfall', 0) < 50:  # Low rainfall
            recommendations['water_conservation'] = {
                'priority': 'high',
                'practices': self.practices['water_conservation'][:3],
                'explanation': 'Low rainfall detected. Focus on water conservation practices.'
            }
        
        # Soil health recommendations
        if soil_conditions.get('organic_matter', 0) < 2.0:  # Low organic matter
            recommendations['soil_health'] = {
                'priority': 'high',
                'practices': self.practices['soil_health'][:3],
                'explanation': 'Low organic matter detected. Improve soil health with organic practices.'
            }
        
        # Nutrient management recommendations
        if (soil_conditions.get('nitrogen', 0) < 20 or 
            soil_conditions.get('phosphorus', 0) < 15 or
            soil_conditions.get('potassium', 0) < 150):
            recommendations['nutrient_management'] = {
                'priority': 'medium',
                'practices': self.practices['nutrient_management'][:3],
                'explanation': 'Nutrient deficiencies detected. Implement proper nutrient management.'
            }
        
        # General recommendations for all crops
        recommendations['general'] = {
            'priority': 'low',
            'practices': [
                'Monitor crop health regularly',
                'Keep detailed records of practices and yields',
                'Stay updated with local agricultural extension services'
            ],
            'explanation': 'General best practices for sustainable farming.'
        }
        
        return recommendations
