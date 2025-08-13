#!/usr/bin/env python3
"""
Enhanced TreeQuest Setup Configuration
======================================

Setup configuration for Enhanced TreeQuest multi-AI provider orchestration system.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="enhanced-treequest",
    version="1.0.0",
    author="Enhanced TreeQuest Contributors",
    author_email="maintainers@enhanced-treequest.dev",
    description="Next-generation multi-AI provider orchestration system with intelligent routing and cost optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enhanced-treequest",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/enhanced-treequest/issues",
        "Documentation": "https://github.com/yourusername/enhanced-treequest/blob/main/README.md",
        "Source Code": "https://github.com/yourusername/enhanced-treequest",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "analytics": [
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "matplotlib>=3.7.0",
        ],
        "monitoring": [
            "prometheus-client>=0.16.0",
            "structlog>=23.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "treequest=enhanced_treequest_controller:main",
            "treequest-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "enhanced_treequest": [
            "*.md",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    keywords=[
        "ai",
        "artificial-intelligence", 
        "orchestration",
        "multi-provider",
        "openai",
        "anthropic",
        "google",
        "cerebras",
        "groq",
        "cost-optimization",
        "parallel-execution",
        "token-tracking",
        "provider-routing",
        "api-management",
    ],
    zip_safe=False,
)