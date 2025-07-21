# Fyndra: An Intelligent Information Retrieval System for E-Commerce Websites

Fyndra is a Python-based information retrieval system designed to crawl, index, and search content from e-commerce websites.

## Setup and Installation

Follow these steps to set up the project environment and install necessary dependencies.

# 1. Create virtual environment

# This command creates an isolated Python environment named 'venv' in the current directory.
# Using a virtual environment helps manage project-specific dependencies without affecting the global Python installation.

python -m venv venv

# 2. Activate the virtual environment

# On Windows (using Command Prompt or PowerShell):
# This command activates the 'venv' virtual environment, making its Python interpreter and installed packages available for use.

venv\Scripts\activate

# On macOS/Linux (using bash/zsh):
# source venv/bin/activate  

# 3. Install required packages

# This command uses pip (Python's package installer) to install all the necessary libraries listed.
# - requests: For making HTTP requests to fetch web pages.
# - beautifulsoup4: For parsing HTML and XML content from web pages.
# - nltk: (Natural Language Toolkit) For text processing tasks like tokenization, stop-word removal, and lemmatization.
# - scikit-learn: For machine learning tasks, including TF-IDF vectorization and potentially PCA.
# - matplotlib: For creating static, animated, and interactive visualizations, such as the PCA scatter plot.

pip install requests beautifulsoup4 nltk scikit-learn matplotlib