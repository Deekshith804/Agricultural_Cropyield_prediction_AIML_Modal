#!/usr/bin/env python3
"""
Comprehensive test script for the Sustainable Agriculture AI/ML System.
"""

import os
import sys
import subprocess
import time
import requests

def test_dependencies():
    """Test that all required dependencies are installed."""
    print("Testing dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'flask', 'dash', 
        'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"[PASS] {package}")
        except ImportError:
            print(f"[FAIL] {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n[FAIL] Missing packages: {', '.join(missing_packages)}")
        return False
    
    print("\n[PASS] All dependencies are installed!")
    return True

def test_data_processing():
    """Test data processing functions."""
    print("\nTesting data processing...")
    
    try:
        # Add utils to path
        sys.path.append('utils')
        from data_processor import load_sample_dataset
        
        # Test loading dataset
        df = load_sample_dataset()
        
        if df.empty:
            print("[FAIL] Failed to load dataset")
            return False
        
        print(f"[PASS] Successfully loaded dataset with {len(df)} records")
        print(f"Dataset shape: {df.shape}")
        
        # Check required columns
        required_columns = [
            'temperature', 'rainfall', 'humidity', 'soil_ph',
            'nitrogen', 'phosphorus', 'potassium', 'organic_matter',
            'irrigation_frequency', 'fertilizer_usage', 'crop_yield'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"[FAIL] Missing required columns: {missing_columns}")
            return False
        
        print("[PASS] All required columns present")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing data processing: {e}")
        return False

def test_model_training():
    """Test model training."""
    print("\nTesting model training...")
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Run training script
        result = subprocess.run([
            sys.executable, "ml_models/train_models.py"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print("[FAIL] Model training failed!")
            print("Error:", result.stderr)
            return False
        
        print("[PASS] Model training completed successfully!")
        
        # Check if model files were created
        model_files = [
            'models/random_forest_model.pkl',
            'models/gradient_boosting_model.pkl'
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        if missing_files:
            print(f"[FAIL] Missing model files: {missing_files}")
            return False
        
        print("[PASS] Model files created successfully")
        return True
        
    except subprocess.TimeoutExpired:
        print("[FAIL] Model training timed out")
        return False
    except Exception as e:
        print(f"[FAIL] Error testing model training: {e}")
        return False

def test_api_server():
    """Test API server."""
    print("\nTesting API server...")
    
    try:
        # Start API server in background
        api_process = subprocess.Popen([
            sys.executable, "api/app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(5)
        
        # Test health endpoint
        response = requests.get("http://localhost:5000/health", timeout=10)
        if response.status_code != 200:
            print("[FAIL] API server health check failed")
            api_process.terminate()
            return False
        
        print("[PASS] API server is running and healthy")
        
        # Test prediction endpoint
        test_data = {
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
        
        response = requests.post(
            "http://localhost:5000/predict/yield", 
            json=test_data, 
            timeout=10
        )
        
        if response.status_code != 200:
            print("[FAIL] API prediction endpoint failed")
            api_process.terminate()
            return False
        
        result = response.json()
        if 'predicted_yield' not in result:
            print("[FAIL] API prediction response missing predicted_yield")
            api_process.terminate()
            return False
        
        print("[PASS] API prediction endpoint working correctly")
        
        # Stop API server
        api_process.terminate()
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing API server: {e}")
        return False

def main():
    """Run all tests."""
    print("Sustainable Agriculture AI/ML System - Comprehensive Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Processing", test_data_processing),
        ("Model Training", test_model_training),
        ("API Server", test_api_server)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} Test {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n[SUCCESS] All tests passed! The system is working correctly.")
        return 0
    else:
        print(f"\n[WARNING] {len(results) - passed} tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
