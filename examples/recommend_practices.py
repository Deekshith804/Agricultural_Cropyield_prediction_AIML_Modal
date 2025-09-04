#!/usr/bin/env python3
"""
Example script showing how to use the sustainable practices recommender.
"""

import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_models.crop_yield_predictor import SustainablePracticeRecommender

def main():
    """Example of using the sustainable practices recommender."""
    print("🌱 Sustainable Agriculture AI/ML System - Practice Recommendation Example")
    print("=" * 70)
    
    # Create recommender
    recommender = SustainablePracticeRecommender()
    
    # Example scenarios
    scenarios = [
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
    
    # Get recommendations for each scenario
    for scenario in scenarios:
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
    main()