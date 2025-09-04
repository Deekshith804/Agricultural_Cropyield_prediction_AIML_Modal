#!/usr/bin/env python3
"""
Sustainable Agriculture AI/ML System - Main Startup Script

This script provides an easy way to run different components of the system.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def print_banner():
    """Print the system banner."""
    print("=" * 70)
    print("🌱 Sustainable Agriculture AI/ML System")
    print("=" * 70)
    print("An intelligent system for crop yield prediction and sustainable practices")
    print("=" * 70)

def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'flask', 'dash', 
        'plotly', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies are installed!")
    return True

def train_models():
    """Train the ML models."""
    print("\n🚀 Training ML models...")
    
    try:
        # Check if models directory exists
        models_dir = Path("models")
        if not models_dir.exists():
            print("Creating models directory...")
            models_dir.mkdir(exist_ok=True)
        
        # Run training script
        result = subprocess.run([
            sys.executable, "ml_models/train_models.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Models trained successfully!")
            print("Models saved in 'models/' directory")
        else:
            print("❌ Model training failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return False
    
    return True

def start_api_server():
    """Start the Flask API server."""
    print("\n🌐 Starting API server...")
    
    try:
        # Start API server in background
        api_process = subprocess.Popen([
            sys.executable, "api/app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:5000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running at http://localhost:5000")
                return api_process
            else:
                print("❌ API server is not responding properly")
                return None
        except:
            print("❌ API server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        return None

def start_dashboard():
    """Start the Dash dashboard."""
    print("\n📊 Starting dashboard...")
    
    try:
        # Start dashboard in background
        dashboard_process = subprocess.Popen([
            sys.executable, "dashboard/app_simple.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for dashboard to start
        time.sleep(5)
        
        print("✅ Dashboard is starting at http://localhost:8050")
        print("Opening dashboard in browser...")
        
        # Open dashboard in browser
        webbrowser.open("http://localhost:8050")
        
        return dashboard_process
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def show_menu():
    """Show the main menu."""
    print("\n" + "=" * 50)
    print("Main Menu")
    print("=" * 50)
    print("1. Train ML Models")
    print("2. Start API Server")
    print("3. Start Dashboard")
    print("4. Start Full System (API + Dashboard)")
    print("5. Check System Status")
    print("6. Exit")
    print("=" * 50)

def check_system_status():
    """Check the status of all system components."""
    print("\n🔍 Checking system status...")
    
    # Check API status
    try:
        import requests
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Server: Running at http://localhost:5000")
        else:
            print("❌ API Server: Not responding properly")
    except:
        print("❌ API Server: Not running")
    
    # Check dashboard status
    try:
        response = requests.get("http://localhost:8050", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard: Running at http://localhost:8050")
        else:
            print("❌ Dashboard: Not responding properly")
    except:
        print("❌ Dashboard: Not running")
    
    # Check models
    models_dir = Path("models")
    if models_dir.exists() and any(models_dir.glob("*.pkl")):
        print("✅ ML Models: Available in 'models/' directory")
    else:
        print("❌ ML Models: Not found - run training first")

def main():
    """Main function."""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ Please install missing dependencies before continuing.")
        return
    
    # Check if models exist
    models_dir = Path("models")
    models_exist = models_dir.exists() and any(models_dir.glob("*.pkl"))
    
    if not models_exist:
        print("\n⚠️  No trained models found. You'll need to train models first.")
        print("Choose option 1 to train models.")
        print("\n💡 Tip: Place your agricultural dataset CSV files in the 'data/' directory")
        print("   before training to use real data instead of generated samples.")
    
    # Main loop
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                if train_models():
                    print("\n✅ Model training completed successfully!")
                    print("💡 Tip: The system automatically uses real datasets from the 'data/' directory")
                    print("   if available, otherwise it generates sample data for demonstration.")
                else:
                    print("\n❌ Model training failed. Check the error messages above.")
            
            elif choice == "2":
                api_process = start_api_server()
                if api_process:
                    print("\n✅ API server started successfully!")
                    print("Press Enter to stop the server...")
                    input()
                    api_process.terminate()
                    print("API server stopped.")
                else:
                    print("\n❌ Failed to start API server.")
            
            elif choice == "3":
                if not models_exist:
                    print("\n❌ No trained models found. Please train models first (option 1).")
                    continue
                
                dashboard_process = start_dashboard()
                if dashboard_process:
                    print("\n✅ Dashboard started successfully!")
                    print("Press Enter to stop the dashboard...")
                    input()
                    dashboard_process.terminate()
                    print("Dashboard stopped.")
                else:
                    print("\n❌ Failed to start dashboard.")
            
            elif choice == "4":
                if not models_exist:
                    print("\n❌ No trained models found. Please train models first (option 1).")
                    continue
                
                print("\n🚀 Starting full system...")
                
                # Start API server
                api_process = start_api_server()
                if not api_process:
                    print("❌ Failed to start API server. Cannot start full system.")
                    continue
                
                # Start dashboard
                dashboard_process = start_dashboard()
                if not dashboard_process:
                    print("❌ Failed to start dashboard. Stopping API server.")
                    api_process.terminate()
                    continue
                
                print("\n✅ Full system is running!")
                print("🌐 API Server: http://localhost:5000")
                print("📊 Dashboard: http://localhost:8050")
                print("\nPress Enter to stop all services...")
                input()
                
                # Stop all services
                api_process.terminate()
                dashboard_process.terminate()
                print("All services stopped.")
            
            elif choice == "5":
                check_system_status()
            
            elif choice == "6":
                print("\n👋 Thank you for using the Sustainable Agriculture AI/ML System!")
                print("Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 6.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
