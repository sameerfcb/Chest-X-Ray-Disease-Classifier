#!/usr/bin/env python
"""Setup script for Chest X-Ray Pneumonia Classifier."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="chest-xray-classifier",
    version="1.2.0",
    author="Sameer Singh",
    author_email="sameer@example.com",
    description="Deep learning model for pneumonia detection in chest X-rays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier",
    project_urls={
        "Bug Tracker": "https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier/issues",
        "Documentation": "https://github.com/sameerfcb/Chest-X-Ray-Disease-Classifier/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "chest-xray-app=src.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
