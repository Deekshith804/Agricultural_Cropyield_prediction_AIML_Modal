#!/usr/bin/env python3
"""
Windows Installation Script for Sustainable Agriculture AI/ML System
This script provides multiple installation methods to resolve dependency issues on Windows.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print the installation banner."""
    print("=" * 70)
    print("🌱 Sustainable Agriculture AI/ML System - Windows Installation")
    print("=" * 70)
    print("This script will help you install dependencies on Windows")
    print("=" * 70)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip is not available")
        return False

def install_with_pip():
    """Install dependencies using pip."""
    print("\n🚀 Installing dependencies with pip...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully")
        
        # Install dependencies
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All dependencies installed successfully!")
            return True
        else:
            print("❌ pip installation failed!")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error during pip installation: {e}")
        return False

def install_with_conda():
    """Install dependencies using conda (if available)."""
    print("\n🚀 Installing dependencies with conda...")
    
    try:
        # Check if conda is available
        result = subprocess.run(["conda", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ conda is not available")
            return False
        
        print("✅ conda is available")
        
        # Create conda environment
        env_name = "sustainable_agriculture"
        subprocess.check_call(["conda", "create", "-n", env_name, "python=3.9", "-y"])
        print(f"✅ Created conda environment: {env_name}")
        
        # Activate environment and install packages
        conda_packages = [
            "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
            "plotly", "flask", "dash", "scipy", "joblib", "requests"
        ]
        
        for package in conda_packages:
            subprocess.check_call(["conda", "install", "-n", env_name, package, "-y"])
            print(f"✅ Installed {package}")
        
        print(f"\n✅ Conda installation completed!")
        print(f"To activate the environment, run: conda activate {env_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error during conda installation: {e}")
        return False

def install_wheels():
    """Install pre-compiled wheels for problematic packages."""
    print("\n🚀 Installing pre-compiled wheels...")
    
    try:
        # Install numpy first (dependency for other packages)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "numpy"
        ])
        print("✅ numpy installed")
        
        # Install scikit-learn from wheel
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "scikit-learn"
        ])
        print("✅ scikit-learn installed")
        
        # Install other packages
        packages = [
            "pandas", "matplotlib", "seaborn", "plotly",
            "flask", "flask-cors", "dash", "dash-bootstrap-components",
            "scipy", "joblib", "python-dotenv", "requests"
        ]
        
        for package in packages:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} installed")
        
        print("✅ All packages installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during wheel installation: {e}")
        return False

def install_visual_cpp():
    """Provide instructions for installing Visual C++ Build Tools."""
    print("\n🔧 Visual C++ Build Tools Installation")
    print("=" * 50)
    print("If you encounter compilation errors, you may need to install")
    print("Microsoft Visual C++ Build Tools.")
    print("\nOptions:")
    print("1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    print("2. Or install via winget: winget install Microsoft.VisualStudio.2022.BuildTools")
    print("3. Or use conda: conda install -c conda-forge m2w64-toolchain")
    print("\nAfter installation, try running the pip install again.")

def test_imports():
    """Test if key packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        'pandas', 'numpy', 'sklearn', 'flask', 'dash', 'plotly'
    ]
    
    failed_imports = []
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All packages imported successfully!")
    return True

def show_menu():
    """Show the installation menu."""
    print("\n" + "=" * 50)
    print("Installation Options")
    print("=" * 50)
    print("1. Install with pip (recommended)")
    print("2. Install with conda (if available)")
    print("3. Install pre-compiled wheels")
    print("4. Install Visual C++ Build Tools (instructions)")
    print("5. Test package imports")
    print("6. Exit")
    print("=" * 50)

def main():
    """Main installation function."""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        return
    
    if not check_pip():
        print("Please install pip first: https://pip.pypa.io/en/stable/installation/")
        return
    
    # Main loop
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                if install_with_pip():
                    print("\n✅ Installation completed successfully!")
                    if test_imports():
                        print("\n🎉 System is ready to use!")
                        print("Run 'python test_system.py' to verify the installation.")
                else:
                    print("\n❌ Installation failed. Try another option.")
            
            elif choice == "2":
                if install_with_conda():
                    print("\n✅ Conda installation completed!")
                else:
                    print("\n❌ Conda installation failed.")
            
            elif choice == "3":
                if install_wheels():
                    print("\n✅ Wheel installation completed!")
                    if test_imports():
                        print("\n🎉 System is ready to use!")
                else:
                    print("\n❌ Wheel installation failed.")
            
            elif choice == "4":
                install_visual_cpp()
            
            elif choice == "5":
                test_imports()
            
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("\n❌ Invalid choice. Please enter a number between 1 and 6.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Installation interrupted.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
