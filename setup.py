#!/usr/bin/env python3
"""
Setup script for cosmos-dataloader package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="cosmos-dataloader",
    version="0.1.0",
    author="Cosmos Team",
    author_email="",
    description="A standalone dataloader with customized webdataset for ML/AI projects",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/cosmos-dataloader",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "cosmos_datasets": ["*.yaml", "*.yml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            "cosmos-convert=examples.convert_webdataset:main",
        ],
    },
    keywords="dataloader, webdataset, machine-learning, computer-vision, video-processing",
    project_urls={
        "Bug Reports": "https://github.com/your-org/cosmos-dataloader/issues",
        "Source": "https://github.com/your-org/cosmos-dataloader",
        "Documentation": "https://cosmos-dataloader.readthedocs.io/",
    },
)
