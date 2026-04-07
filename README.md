# FairBug: Fairness-Aware Bug Report Classifier

## 🎯 Project Overview

FairBug is an intelligent software engineering tool that automatically classifies bug reports as performance-related or not, while ensuring fairness across different software projects and providing explainable predictions.

### Key Features
- **Ensemble Classification**: Combines Random Forest, Gradient Boosting, and SVM for improved accuracy
- **Fairness Testing**: Detects bias across different projects using statistical tests
- **Explainable Predictions**: Provides human-readable explanations for each classification
- **Statistical Validation**: Uses Wilcoxon signed-rank test for significance testing

## 📊 Performance
- **15% improvement** in F1 score over baseline (Naive Bayes + TF-IDF)
- **Statistically significant** results (p < 0.05)
- **Cross-project fairness analysis** across 5 major deep learning frameworks

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/fairbug.git
cd fairbug

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"