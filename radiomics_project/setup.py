from setuptools import setup, find_packages

setup(
    name="radiomics_project",
    version="1.0.0",
    author="Medical AI Team",
    author_email="medical.ai@example.com",
    description="Medical imaging radiomics pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/radiomics_project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "SimpleITK>=2.3.0",
        "pyradiomics>=3.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "torch>=2.0.0",
        "monai>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "radiomics-pipeline=main:main",
        ],
    },
)
