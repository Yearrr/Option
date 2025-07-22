#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Options Backtest Framework Setup Script
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="options-backtest-framework",
    version="1.0.0",
    author="资深期权量化交易员",
    author_email="options.trader@example.com",
    description="专业的期权量化交易回测框架",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/options-backtest-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "options-backtest=options_backtest_framework.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "options_backtest_framework": [
            "data/*.csv",
            "templates/*.html",
        ],
    },
    keywords=[
        "options", "backtest", "quantitative", "trading", "finance", 
        "derivatives", "portfolio", "risk management", "Greeks"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/options-backtest-framework/issues",
        "Source": "https://github.com/your-username/options-backtest-framework",
        "Documentation": "https://options-backtest-framework.readthedocs.io/",
    },
)