# setup.py

from setuptools import setup, find_packages

setup(
    name='style2speed',
    version='0.1',
    description='Neural network pipeline to analyze F1 telemetry and infer driver style and performance differences.',
    author='Roberta Poceviciute',
    packages=find_packages(),  # Automatically finds 'style2speed' and submodules
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "torch",
        "tqdm",
        "fastf1",
        "captum",
    ],
    python_requires='>=3.8',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)