from setuptools import setup, find_packages

setup(
    name="ml-audit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn"
    ],
    extras_require={
        'balance': ['imbalanced-learn']
    },
    author="ML Audit Team",
    description="A lightweight library for tracking, auditing, and visualizing machine learning preprocessing steps.",
    long_description=open("README.md").read() if open("README.md").read else "",
    long_description_content_type="text/markdown",
)
