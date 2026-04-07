"""
Setup script for FairBug
"""

from setuptools import setup, find_packages

setup(
    name='fairbug',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@university.ac.uk',
    description='Fairness-Aware Bug Report Classifier',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'scipy>=1.10.0',
        'nltk>=3.8.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'tqdm>=4.65.0',
        'eli5>=0.11.0',
        'shap>=0.42.0',
        'statsmodels>=0.14.0',
        'joblib>=1.3.0'
    ],
    python_requires='>=3.8',
)