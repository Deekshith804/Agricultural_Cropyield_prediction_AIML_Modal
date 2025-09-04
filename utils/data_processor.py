import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgriculturalDataProcessor:
    """
    Utility class for processing and validating agricultural data.
    """
    
    def __init__(self):
        # Define optimal ranges for different parameters
        self.optimal_ranges = {
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
        
        # Crop-specific requirements
        self.crop_requirements = {
            'wheat': {
                'optimal_temp': 20,
                'optimal_ph': 6.5,
                'water_needs': 'moderate',
                'nutrient_needs': 'high'
            },
            'corn': {
                'optimal_temp': 25,
                'optimal_ph': 6.0,
                'water_needs': 'high',
                'nutrient_needs': 'very_high'
            },
            'soybeans': {
                'optimal_temp': 22,
                'optimal_ph': 6.8,
                'water_needs': 'moderate',
                'nutrient_needs': 'medium'
            },
            'rice': {
                'optimal_temp': 28,
                'optimal_ph': 6.0,
                'water_needs': 'very_high',
                'nutrient_needs': 'high'
            }
        }
    
    def validate_input_data(self, data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate input data for crop yield prediction.
        
        Args:
            data: Dictionary containing input parameters
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        required_fields = [
            'temperature', 'rainfall', 'humidity', 'soil_ph',
            'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
            'irrigation_frequency', 'fertilizer_usage'
        ]
        
        # Check for missing fields
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
                continue
            
            value = data[field]
            
            # Check data types
            if not isinstance(value, (int, float)):
                errors.append(f"Field {field} must be numeric, got {type(value)}")
                continue
            
            # Check ranges
            if field in self.optimal_ranges:
                range_info = self.optimal_ranges[field]
                if value < range_info['min'] or value > range_info['max']:
                    errors.append(
                        f"Field {field} value {value} is outside recommended range "
                        f"({range_info['min']}-{range_info['max']} {range_info['unit']})"
                    )
        
        return len(errors) == 0, errors
    
    def calculate_soil_health_score(self, soil_data: Dict) -> Dict:
        """
        Calculate a soil health score based on multiple parameters.
        
        Args:
            soil_data: Dictionary containing soil parameters
            
        Returns:
            Dictionary with soil health score and breakdown
        """
        scores = {}
        total_score = 0
        max_possible_score = 0
        
        # pH score (optimal around 6.5)
        if 'soil_ph' in soil_data:
            ph = soil_data['soil_ph']
            ph_score = max(0, 100 - abs(ph - 6.5) * 20)
            scores['ph_score'] = ph_score
            total_score += ph_score
            max_possible_score += 100
        
        # Organic matter score
        if 'organic_matter' in soil_data:
            om = soil_data['organic_matter']
            om_score = min(100, om * 25)  # 4% = 100 points
            scores['organic_matter_score'] = om_score
            total_score += om_score
            max_possible_score += 100
        
        # Nutrient balance score
        if all(key in soil_data for key in ['nitrogen', 'phosphorus', 'potassium']):
            n, p, k = soil_data['nitrogen'], soil_data['phosphorus'], soil_data['potassium']
            
            # Calculate nutrient ratios
            n_p_ratio = n / p if p > 0 else 0
            k_ratio = k / 200  # Normalize to optimal K level
            
            # Score based on optimal ratios
            ratio_score = max(0, 100 - abs(n_p_ratio - 1.5) * 30)
            k_score = min(100, k_ratio * 100)
            
            scores['nutrient_balance_score'] = (ratio_score + k_score) / 2
            total_score += scores['nutrient_balance_score']
            max_possible_score += 100
        
        # Overall soil health score
        overall_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0
        
        return {
            'overall_score': round(overall_score, 1),
            'component_scores': scores,
            'max_possible_score': max_possible_score,
            'total_score': total_score
        }
    
    def analyze_weather_conditions(self, weather_data: Dict) -> Dict:
        """
        Analyze weather conditions for agricultural suitability.
        
        Args:
            weather_data: Dictionary containing weather parameters
            
        Returns:
            Dictionary with weather analysis and recommendations
        """
        analysis = {
            'suitability_score': 0,
            'risks': [],
            'recommendations': [],
            'seasonal_considerations': []
        }
        
        # Temperature analysis
        if 'temperature' in weather_data:
            temp = weather_data['temperature']
            if temp < 10:
                analysis['risks'].append('Temperature too low for most crops')
                analysis['recommendations'].append('Consider cold-tolerant crops or wait for warmer weather')
            elif temp > 35:
                analysis['risks'].append('Temperature too high, risk of heat stress')
                analysis['recommendations'].append('Implement shade structures and increase irrigation')
            else:
                analysis['suitability_score'] += 25
        
        # Rainfall analysis
        if 'rainfall' in weather_data:
            rainfall = weather_data['rainfall']
            if rainfall < 30:
                analysis['risks'].append('Low rainfall, drought conditions likely')
                analysis['recommendations'].append('Implement water conservation practices and drought-resistant crops')
            elif rainfall > 200:
                analysis['risks'].append('Excessive rainfall, risk of waterlogging')
                analysis['recommendations'].append('Ensure proper drainage and avoid heavy machinery on wet soil')
            else:
                analysis['suitability_score'] += 25
        
        # Humidity analysis
        if 'humidity' in weather_data:
            humidity = weather_data['humidity']
            if humidity > 85:
                analysis['risks'].append('High humidity, increased disease risk')
                analysis['recommendations'].append('Improve air circulation and monitor for fungal diseases')
            elif humidity < 30:
                analysis['risks'].append('Low humidity, increased water loss')
                analysis['recommendations'].append('Increase irrigation frequency and use mulch')
            else:
                analysis['suitability_score'] += 25
        
        # Seasonal considerations
        current_month = pd.Timestamp.now().month
        if current_month in [3, 4, 5]:  # Spring
            analysis['seasonal_considerations'].append('Spring planting season - prepare soil and plan crops')
        elif current_month in [6, 7, 8]:  # Summer
            analysis['seasonal_considerations'].append('Summer growing season - monitor water needs and pest pressure')
        elif current_month in [9, 10, 11]:  # Fall
            analysis['seasonal_considerations'].append('Fall harvest season - plan harvest timing and storage')
        else:  # Winter
            analysis['seasonal_considerations'].append('Winter planning season - prepare for next growing season')
        
        return analysis
    
    def generate_sustainability_report(self, soil_data: Dict, weather_data: Dict, 
                                    crop_type: str, predicted_yield: float) -> Dict:
        """
        Generate a comprehensive sustainability report.
        
        Args:
            soil_data: Soil condition data
            weather_data: Weather condition data
            crop_type: Type of crop being grown
            predicted_yield: Predicted crop yield
            
        Returns:
            Dictionary containing sustainability report
        """
        # Calculate scores
        soil_health = self.calculate_soil_health_score(soil_data)
        weather_analysis = self.analyze_weather_conditions(weather_data)
        
        # Sustainability indicators
        sustainability_score = (soil_health['overall_score'] + weather_analysis['suitability_score']) / 2
        
        # Resource efficiency score (simplified)
        resource_efficiency = 0
        if 'fertilizer_usage' in soil_data:
            fertilizer = soil_data['fertilizer_usage']
            if fertilizer <= 100:
                resource_efficiency += 50
            elif fertilizer <= 150:
                resource_efficiency += 25
        
        if 'irrigation_frequency' in soil_data:
            irrigation = soil_data['irrigation_frequency']
            if irrigation <= 3:
                resource_efficiency += 50
            elif irrigation <= 4:
                resource_efficiency += 25
        
        # Overall sustainability score
        overall_sustainability = (sustainability_score + resource_efficiency) / 2
        
        return {
            'overall_sustainability_score': round(overall_sustainability, 1),
            'soil_health': soil_health,
            'weather_analysis': weather_analysis,
            'resource_efficiency': resource_efficiency,
            'crop_specific_insights': self.crop_requirements.get(crop_type, {}),
            'recommendations': {
                'immediate_actions': self._get_immediate_actions(soil_data, weather_data),
                'long_term_improvements': self._get_long_term_improvements(soil_data),
                'resource_optimization': self._get_resource_optimization(soil_data)
            }
        }
    
    def _get_immediate_actions(self, soil_data: Dict, weather_data: Dict) -> List[str]:
        """Get immediate actions based on current conditions."""
        actions = []
        
        if soil_data.get('soil_ph', 7) < 6.0:
            actions.append('Apply lime to raise soil pH')
        
        if soil_data.get('organic_matter', 0) < 2.0:
            actions.append('Add organic matter (compost, manure)')
        
        if weather_data.get('rainfall', 100) < 50:
            actions.append('Increase irrigation frequency')
        
        return actions
    
    def _get_long_term_improvements(self, soil_data: Dict) -> List[str]:
        """Get long-term improvement recommendations."""
        improvements = []
        
        improvements.append('Implement crop rotation system')
        improvements.append('Establish cover cropping program')
        improvements.append('Develop soil testing schedule')
        improvements.append('Plan organic matter addition program')
        
        return improvements
    
    def _get_resource_optimization(self, soil_data: Dict) -> List[str]:
        """Get resource optimization recommendations."""
        optimizations = []
        
        if soil_data.get('fertilizer_usage', 100) > 150:
            optimizations.append('Reduce fertilizer usage through precision agriculture')
        
        if soil_data.get('irrigation_frequency', 3) > 4:
            optimizations.append('Optimize irrigation schedule using soil moisture sensors')
        
        optimizations.append('Implement integrated pest management')
        optimizations.append('Use slow-release fertilizers')
        
        return optimizations

def load_sample_dataset() -> pd.DataFrame:
    """
    Load or generate a sample agricultural dataset for demonstration.
    
    Returns:
        DataFrame with sample agricultural data
    """
    # Try to load real datasets first
    data_dir = 'data'
    if os.path.exists(data_dir):
        # Try loading FAO dataset
        fao_files = [f for f in os.listdir(data_dir) if 'fao' in f.lower() and f.endswith('.csv')]
        if fao_files:
            df = load_fao_dataset(os.path.join(data_dir, fao_files[0]))
            if not df.empty:
                logger.info("Loaded FAO dataset")
                return df
        
        # Try loading India agriculture dataset
        india_files = [f for f in os.listdir(data_dir) if ('india' in f.lower() or 'indian' in f.lower()) and f.endswith('.csv')]
        if india_files:
            df = load_india_agriculture_dataset(os.path.join(data_dir, india_files[0]))
            if not df.empty:
                logger.info("Loaded India agriculture dataset")
                return df
        
        # Try loading Kaggle dataset
        kaggle_files = [f for f in os.listdir(data_dir) if 'kaggle' in f.lower() and f.endswith('.csv')]
        if kaggle_files:
            df = load_kaggle_crop_dataset(os.path.join(data_dir, kaggle_files[0]))
            if not df.empty:
                logger.info("Loaded Kaggle crop dataset")
                return df
        
        # Try loading generic agricultural dataset
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        for csv_file in csv_files:
            # Skip the sample file we might have created
            if 'sample' in csv_file.lower():
                continue
            df = load_agricultural_dataset(os.path.join(data_dir, csv_file))
            if not df.empty:
                logger.info(f"Loaded agricultural dataset from {csv_file}")
                return df
    
    # Try to load sample data file
    try:
        df = pd.read_csv('data/sample_agricultural_data.csv')
        logger.info("Loaded sample dataset from file")
        return df
    except FileNotFoundError:
        # Generate sample data if no real datasets found
        logger.info("Generating sample dataset")
        
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'temperature': np.random.normal(25, 8, n_samples),
            'rainfall': np.random.exponential(50, n_samples),
            'humidity': np.random.uniform(40, 90, n_samples),
            'soil_ph': np.random.normal(6.5, 1.0, n_samples),
            'nitrogen': np.random.normal(30, 10, n_samples),
            'phosphorus': np.random.normal(20, 8, n_samples),
            'potassium': np.random.normal(200, 50, n_samples),
            'organic_matter': np.random.normal(3.0, 1.0, n_samples),
            'irrigation_frequency': np.random.poisson(3, n_samples),
            'fertilizer_usage': np.random.exponential(100, n_samples),
            'crop_type': np.random.choice(['wheat', 'corn', 'soybeans', 'rice'], n_samples)
        }
        
        # Create target variable
        base_yield = 5.0
        temp_effect = -0.1 * (data['temperature'] - 25)**2 + 2.0
        rain_effect = -0.0001 * (data['rainfall'] - 100)**2 + 1.5
        ph_effect = -0.5 * (data['soil_ph'] - 6.5)**2 + 1.0
        nutrient_effect = (0.02 * data['nitrogen'] + 0.03 * data['phosphorus'] + 0.001 * data['potassium']) / 100
        om_effect = 0.3 * data['organic_matter']
        irrigation_effect = 0.1 * data['irrigation_frequency']
        fertilizer_effect = 0.5 * (1 - np.exp(-0.01 * data['fertilizer_usage']))
        
        data['crop_yield'] = np.maximum(0, base_yield + temp_effect + rain_effect +
                                       ph_effect + nutrient_effect + om_effect +
                                       irrigation_effect + fertilizer_effect +
                                       np.random.normal(0, 0.5, n_samples))
        
        df = pd.DataFrame(data)
        
        # Save to file for future use
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_agricultural_data.csv', index=False)
        logger.info("Saved sample dataset to data/sample_agricultural_data.csv")
        
        return df

def export_data_to_json(data: Dict, filename: str) -> bool:
    """
    Export data to JSON file.
    
    Args:
        data: Data to export
        filename: Output filename
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Data exported to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
def load_fao_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess FAO crop yield dataset.
    
    Args:
        filepath: Path to the FAO dataset CSV file
        
    Returns:
        DataFrame with preprocessed data in the standard format
    """
    try:
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # FAO datasets typically have columns like:
        # Area, Item, Year, Unit, Value
        # We need to map these to our standard format
        
        # Filter for crop yield data (if not already filtered)
        yield_data = df[df['Item'].str.contains('yield', case=False, na=False)]
        
        # For demonstration, we'll convert a simplified FAO format
        # In practice, you would need to adjust this based on the actual FAO dataset structure
        processed_data = pd.DataFrame({
            'temperature': np.random.normal(25, 8, len(yield_data)),  # Placeholder
            'rainfall': np.random.exponential(50, len(yield_data)),   # Placeholder
            'humidity': np.random.uniform(40, 90, len(yield_data)),   # Placeholder
            'soil_ph': np.random.normal(6.5, 1.0, len(yield_data)),  # Placeholder
            'nitrogen': np.random.normal(30, 10, len(yield_data)),    # Placeholder
            'phosphorus': np.random.normal(20, 8, len(yield_data)),   # Placeholder
            'potassium': np.random.normal(200, 50, len(yield_data)),  # Placeholder
            'organic_matter': np.random.normal(3.0, 1.0, len(yield_data)),  # Placeholder
            'irrigation_frequency': np.random.poisson(3, len(yield_data)),   # Placeholder
            'fertilizer_usage': np.random.exponential(100, len(yield_data)), # Placeholder
            'crop_type': yield_data['Item'].str.lower(),
            'crop_yield': yield_data['Value']
        })
        
        logger.info(f"Loaded and preprocessed FAO dataset with {len(processed_data)} records")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading FAO dataset: {e}")
        return pd.DataFrame()

def load_kaggle_crop_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess Kaggle crop production dataset.
    
    Args:
        filepath: Path to the Kaggle dataset CSV file
        
    Returns:
        DataFrame with preprocessed data in the standard format
    """
    try:
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # Kaggle crop datasets can have various formats
        # This is a generic preprocessing function that would need to be
        # customized based on the specific Kaggle dataset being used
        
        # Map columns to our standard format (this is a placeholder implementation)
        processed_data = pd.DataFrame({
            'temperature': df.get('temperature', np.random.normal(25, 8, len(df))),
            'rainfall': df.get('rainfall', np.random.exponential(50, len(df))),
            'humidity': df.get('humidity', np.random.uniform(40, 90, len(df))),
            'soil_ph': df.get('soil_ph', np.random.normal(6.5, 1.0, len(df))),
            'nitrogen': df.get('nitrogen', np.random.normal(30, 10, len(df))),
            'phosphorus': df.get('phosphorus', np.random.normal(20, 8, len(df))),
            'potassium': df.get('potassium', np.random.normal(200, 50, len(df))),
            'organic_matter': df.get('organic_matter', np.random.normal(3.0, 1.0, len(df))),
            'irrigation_frequency': df.get('irrigation_frequency', np.random.poisson(3, len(df))),
            'fertilizer_usage': df.get('fertilizer_usage', np.random.exponential(100, len(df))),
            'crop_type': df.get('crop', df.get('crop_type', 'wheat')),
            'crop_yield': df.get('yield', df.get('production', np.random.normal(5.0, 1.0, len(df))))
        })
        
        logger.info(f"Loaded and preprocessed Kaggle crop dataset with {len(processed_data)} records")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading Kaggle crop dataset: {e}")
        return pd.DataFrame()

def load_india_agriculture_dataset(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess India government agriculture dataset.
    
    Args:
        filepath: Path to the India agriculture dataset CSV file
        
    Returns:
        DataFrame with preprocessed data in the standard format
    """
    try:
        # Load the dataset
        df = pd.read_csv(filepath)
        
        # India agriculture datasets typically have columns like:
        # State, District, Crop, Production, Area, Season, etc.
        # We need to map these to our standard format
        
        # For demonstration, we'll convert a simplified format
        # In practice, you would need to adjust this based on the actual dataset structure
        processed_data = pd.DataFrame({
            'temperature': np.random.normal(25, 8, len(df)),  # Placeholder
            'rainfall': np.random.exponential(50, len(df)),   # Placeholder
            'humidity': np.random.uniform(40, 90, len(df)),   # Placeholder
            'soil_ph': np.random.normal(6.5, 1.0, len(df)),  # Placeholder
            'nitrogen': np.random.normal(30, 10, len(df)),    # Placeholder
            'phosphorus': np.random.normal(20, 8, len(df)),   # Placeholder
            'potassium': np.random.normal(200, 50, len(df)),  # Placeholder
            'organic_matter': np.random.normal(3.0, 1.0, len(df)),  # Placeholder
            'irrigation_frequency': np.random.poisson(3, len(df)),   # Placeholder
            'fertilizer_usage': np.random.exponential(100, len(df)), # Placeholder
            'crop_type': df.get('Crop', 'wheat'),
            'crop_yield': df.get('Production', df.get('Yield', np.random.normal(5.0, 1.0, len(df))))
        })
        
        logger.info(f"Loaded and preprocessed India agriculture dataset with {len(processed_data)} records")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading India agriculture dataset: {e}")
        return pd.DataFrame()
        return False

def load_agricultural_dataset(filepath: str, dataset_type: str = 'auto') -> pd.DataFrame:
    """
    Load and preprocess agricultural dataset with automatic format detection.
    
    Args:
        filepath: Path to the dataset file
        dataset_type: Type of dataset ('fao', 'kaggle', 'india', or 'auto' for automatic detection)
        
    Returns:
        DataFrame with preprocessed data in the standard format
    """
    try:
        # Determine dataset type if set to auto
        if dataset_type == 'auto':
            filename = filepath.lower()
            if 'fao' in filename or 'food_and_agriculture' in filename:
                dataset_type = 'fao'
            elif 'kaggle' in filename:
                dataset_type = 'kaggle'
            elif 'india' in filename or 'indian' in filename:
                dataset_type = 'india'
            else:
                # Try to detect based on file content
                df = pd.read_csv(filepath, nrows=5)  # Read first 5 rows for detection
                if 'Area' in df.columns and 'Item' in df.columns and 'Value' in df.columns:
                    dataset_type = 'fao'
                elif 'State' in df.columns and 'District' in df.columns and 'Crop' in df.columns:
                    dataset_type = 'india'
                else:
                    dataset_type = 'kaggle'  # Default to Kaggle format
        
        # Load dataset based on detected type
        if dataset_type == 'fao':
            return load_fao_dataset(filepath)
        elif dataset_type == 'india':
            return load_india_agriculture_dataset(filepath)
        else:  # Default to Kaggle
            return load_kaggle_crop_dataset(filepath)
            
    except Exception as e:
        logger.error(f"Error loading agricultural dataset: {e}")
        return pd.DataFrame()
