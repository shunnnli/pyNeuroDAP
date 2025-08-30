#!/usr/bin/env python3
"""
Simple script to upload pyNeuroDAP to PyPI
Usage: python upload_to_pypi.py [version]
"""

import os
import sys
import subprocess
import re
from pathlib import Path

def update_version_files(new_version):
    """Update version in all relevant files"""
    print(f"Updating version to {new_version}...")
    
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    if pyproject_path.exists():
        with open(pyproject_path, 'r') as f:
            content = f.read()
        content = re.sub(r'version = "[\d.]+"', f'version = "{new_version}"', content)
        with open(pyproject_path, 'w') as f:
            f.write(content)
        print("✓ Updated pyproject.toml")
    
    # Update __init__.py
    init_path = Path("pyNeuroDAP/__init__.py")
    if init_path.exists():
        with open(init_path, 'r') as f:
            content = f.read()
        content = re.sub(r'__version__ = "[\d.]+"', f'__version__ = "{new_version}"', content)
        with open(init_path, 'w') as f:
            f.write(content)
        print("✓ Updated pyNeuroDAP/__init__.py")
    
    # Update setup.py
    setup_path = Path("setup.py")
    if setup_path.exists():
        with open(setup_path, 'r') as f:
            content = f.read()
        content = re.sub(r'version="[\d.]+"', f'version="{new_version}"', content)
        with open(setup_path, 'w') as f:
            f.write(content)
        print("✓ Updated setup.py")

def clean_and_build():
    """Clean old builds and create new distribution"""
    print("Cleaning old builds...")
    subprocess.run(["rm", "-rf", "dist/"], check=True)
    subprocess.run(["rm", "-rf", "build/"], check=True)
    subprocess.run(["rm", "-rf", "*.egg-info/"], check=True)
    
    print("Building new distribution...")
    result = subprocess.run(["python", "-m", "build"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Build failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    print("✓ Build successful")

def upload_to_pypi():
    """Upload to PyPI"""
    print("Uploading to PyPI...")
    result = subprocess.run(["python", "-m", "twine", "upload", "dist/*"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Upload failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    print("✓ Upload successful!")

def main():
    if len(sys.argv) != 2:
        print("Usage: python upload_to_pypi.py <version>")
        print("Example: python upload_to_pypi.py 0.1.1")
        sys.exit(1)
    
    new_version = sys.argv[1]
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+$', new_version):
        print("❌ Invalid version format. Use format: X.Y.Z (e.g., 0.1.1)")
        sys.exit(1)
    
    print(f"🚀 Starting PyPI upload for version {new_version}")
    print("=" * 50)
    
    try:
        # Step 1: Update version files
        update_version_files(new_version)
        
        # Step 2: Clean and build
        clean_and_build()
        
        # Step 3: Upload to PyPI
        upload_to_pypi()
        
        print("=" * 50)
        print(f"🎉 Successfully uploaded pyNeuroDAP v{new_version} to PyPI!")
        print(f"📦 Package available at: https://pypi.org/project/pyNeuroDAP/{new_version}/")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
