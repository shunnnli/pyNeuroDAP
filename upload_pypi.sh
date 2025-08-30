#!/bin/bash

# Simple shell script to upload pyNeuroDAP to PyPI
# Usage: ./upload_pypi.sh <version>

if [ $# -eq 0 ]; then
    echo "Usage: ./upload_pypi.sh <version>"
    echo "Example: ./upload_pypi.sh 0.1.1"
    exit 1
fi

VERSION=$1

# Activate conda environment
echo "Activating pyNeuroDAP conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pyNeuroDAP

# Run the Python upload script
echo "Running upload script..."
python upload_to_pypi.py $VERSION

echo "Done!"
