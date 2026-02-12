from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forgenn",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modern neural network framework built from scratch with NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/forgenn",
    py_modules=["forgenn"],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
        ],
    },
    keywords="deep-learning neural-network machine-learning ai transformer resnet numpy",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/forgenn/issues",
        "Source": "https://github.com/yourusername/forgenn",
        "Documentation": "https://github.com/yourusername/forgenn#readme",
    },
)
