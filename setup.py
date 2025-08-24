from setuptools import setup, find_packages

setup(
    name="medigen-ai",
    version="1.0.0",
    description="AI-powered healthcare companion for symptom analysis and health recommendations",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "flask>=2.3.3",
        "transformers>=4.35.0",
        "tensorflow>=2.13.0",
        "pandas>=2.1.1",
        "numpy>=1.24.3",
        "matplotlib>=3.7.2",
        "seaborn>=0.12.2",
        "scikit-learn>=1.3.0",
        "torch>=2.0.1",
        "plotly>=5.17.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
