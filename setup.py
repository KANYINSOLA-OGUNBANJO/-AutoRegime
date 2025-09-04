Copyfrom setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="autoregime",
    version="1.0.0",
    author="Kanyinsola Ogunbanjo",
    author_email="kanyinsolaogunbanjo@gmail.com",
    description="Professional market regime detection system for research and analysis purposes",
    long_description=long_description + "\n\n**Disclaimer:** This tool is for research and analysis purposes only. Past performance does not guarantee future results. Not financial advice.",
    long_description_content_type="text/markdown",
    url="https://github.com/KANYINSOLA-OGUNBANJO/-AutoRegime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "hmmlearn>=0.2.7",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "streamlit>=1.10.0",
        "yfinance>=0.1.87",
        "seaborn>=0.11.0",
    ],
    keywords=[
        "market regime detection",
        "financial analysis", 
        "hidden markov models",
        "quantitative finance",
        "research tools"
    ],
)
