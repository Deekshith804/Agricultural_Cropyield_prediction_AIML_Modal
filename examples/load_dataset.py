#!/usr/bin/env python3
"""
Example script showing how to load and preprocess agricultural datasets.
"""

import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_processor import (
    load_sample_dataset, 
    load_fao_dataset, 
    load_kaggle_crop_dataset, 
    load_india_agriculture_dataset,
    load_agricultural_dataset
)
import pandas as pd

def main():
    """Example of loading agricultural datasets."""
    print("🌱 Sustainable Agriculture AI/ML System - Dataset Loading Example")
    print("=" * 65)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Loading sample dataset...")
    try:
        # This will try to load real datasets first, then generate sample data
        df = load_sample_dataset()
        
        if not df.empty:
            print(f"✅ Successfully loaded dataset with {len(df)} records")
            print(f"Dataset shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())
            
            print("\nColumn names:")
            for col in df.columns:
                print(f"  - {col}")
                
            print("\nBasic statistics:")
            print(df.describe())
        else:
            print("❌ Failed to load dataset")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
    
    print("\n" + "=" * 50)
    print("Examples of loading specific dataset types:")
    print("=" * 50)
    
    # Examples of loading specific dataset types
    print("\n1. Loading FAO dataset:")
    print("   df = load_fao_dataset('data/fao_dataset.csv')")
    
    print("\n2. Loading Kaggle crop dataset:")
    print("   df = load_kaggle_crop_dataset('data/kaggle_crop_data.csv')")
    
    print("\n3. Loading India agriculture dataset:")
    print("   df = load_india_agriculture_dataset('data/india_agriculture.csv')")
    
    print("\n4. Loading dataset with automatic detection:")
    print("   df = load_agricultural_dataset('data/my_dataset.csv')")
    
    print("\n💡 Tip: Place your CSV files in the 'data/' directory")
    print("   and the system will automatically detect and use them!")

if __name__ == "__main__":
    main()