"""
fishstick - Package Configuration
"""

from setuptools import setup, find_packages

setup(
    name="fishstick",
    version="0.1.0",
    description="A mathematically rigorous, physically grounded AI framework",
    author="NeuralBlitz",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.21",
        "scipy>=1.7",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "mypy>=1.0",
        ],
        "full": [
            "gudhi>=3.8",
            "giotto-tda>=0.6",
            "kmapper>=2.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
