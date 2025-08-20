#!/usr/bin/env python3
"""
Simple installation script for NeuroDAP package
Run this script to install the package in development mode
"""

import subprocess
import sys
import os

def install_package():
    """Install the NeuroDAP package in development mode"""
    print("🧠 Installing NeuroDAP package...")
    
    try:
        # Check if we're in the right directory
        if not os.path.exists('setup.py'):
            print("❌ Error: setup.py not found. Please run this script from the package root directory.")
            return False
        
        # Install in development mode
        print("📦 Installing package in development mode...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Package installed successfully!")
            print("\n🎉 You can now use:")
            print("  import pyNeuroDAP as ndap")
            print("  ndap.get_spikes(...)")
            print("  ndap.save_session_data(...)")
            return True
        else:
            print("❌ Installation failed!")
            print("Error output:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def test_installation():
    """Test if the package can be imported after installation"""
    print("\n🧪 Testing installation...")
    
    try:
        import pyNeuroDAP as ndap
        print("✅ Package imported successfully!")
        print(f"✅ Version: {ndap.__version__}")
        print(f"✅ Author: {ndap.__author__}")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Main installation function"""
    print("=" * 50)
    print("🧠 NeuroDAP Package Installer")
    print("=" * 50)
    
    # Install the package
    if install_package():
        # Test the installation
        if test_installation():
            print("\n🎉 Installation completed successfully!")
            print("\n📚 For usage examples, see README.md")
            print("🔧 For development, the package is installed in editable mode")
        else:
            print("\n⚠️  Package installed but import test failed")
    else:
        print("\n❌ Installation failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
