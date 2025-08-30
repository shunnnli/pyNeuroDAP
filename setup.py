from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pyNeuroDAP",
    version="1.1.0",
    author="Shun Li",
    author_email="shunli@g.harvard.edu",  # Add your email if desired
    description="Neural Data Analysis Package for spike processing, trial management, and modeling",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/shunnnli/pyNeuroDAP",  # Add your repository URL if desired
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "h5py>=3.0.0",
        "tables>=3.7.0",
        "tqdm>=4.60.0",
        "dask>=2022.0.0",
        "PySide6>=6.0.0",
        "pathlib2>=2.3.0; python_version<'3.4'",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "numpydoc>=1.1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="neuroscience, neural data, spike analysis, electrophysiology, rSLDS",
    project_urls={
        "Bug Reports": "",  # Add your issue tracker URL
        "Source": "",        # Add your repository URL
    },
)
