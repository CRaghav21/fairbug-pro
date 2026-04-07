"""
Utility functions for FairBug project
"""

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import warnings
warnings.filterwarnings('ignore')

def preprocess_text(text):
    """Preprocess bug report text"""
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

def load_dataset(project_name, data_path='data/raw'):
    """
    Load dataset for a specific project
    """
    filepath = f'{data_path}/{project_name}_reports.csv'
    
    print(f"Attempting to load: {filepath}")
    
    # Try to load real data
    if os.path.exists(filepath):
        try:
            print(f"Loading REAL data from {filepath}...")
            df = pd.read_csv(filepath)
            
            print(f"Columns found: {list(df.columns)}")
            print(f"Total rows: {len(df)}")
            
            # Check what columns are available
            # The real data might have different column names
            possible_text_columns = ['description', 'text', 'body', 'content']
            possible_label_columns = ['label', 'is_performance', 'type']
            
            # Find text column
            text_col = None
            for col in possible_text_columns:
                if col in df.columns:
                    text_col = col
                    break
            
            # Find label column
            label_col = None
            for col in possible_label_columns:
                if col in df.columns:
                    label_col = col
                    break
            
            if text_col is None or label_col is None:
                print(f"⚠ Could not find text/label columns in {filepath}")
                print(f"Available columns: {list(df.columns)}")
                print("Falling back to synthetic data...")
                return create_synthetic_data(project_name)
            
            # Get texts and labels
            texts = df[text_col].fillna('').astype(str).values
            labels = df[label_col].values
            
            print(f"Original text sample: {texts[0][:100]}...")
            
            # Preprocess texts
            texts = [preprocess_text(t) for t in texts]
            
            print(f"Preprocessed sample: {texts[0][:100]}...")
            print(f"✓ Loaded {len(texts)} samples with {sum(labels)} performance bugs ({sum(labels)/len(labels)*100:.1f}%)")
            
            return np.array(texts), np.array(labels)
            
        except Exception as e:
            print(f"⚠ Error loading {filepath}: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to synthetic data...")
    
    # Fall back to synthetic data
    return create_synthetic_data(project_name)

def create_synthetic_data(project_name):
    """Create synthetic data for testing"""
    np.random.seed(42)
    
    # Sample sizes based on project
    sizes = {
        'tensorflow': 1490,
        'pytorch': 752,
        'keras': 668,
        'mxnet': 516,
        'caffe': 286
    }
    
    # Performance ratios
    ratios = {
        'tensorflow': 0.187,
        'pytorch': 0.126,
        'keras': 0.202,
        'mxnet': 0.126,
        'caffe': 0.115
    }
    
    n_samples = sizes.get(project_name, 500)
    perf_ratio = ratios.get(project_name, 0.16)
    n_perf = int(n_samples * perf_ratio)
    n_non_perf = n_samples - n_perf
    
    # Keywords
    perf_keywords = ['slow', 'memory', 'performance', 'crash', 'timeout', 'speed', 'gpu', 'cpu', 'latency', 'bottleneck']
    other_keywords = ['feature', 'documentation', 'install', 'setup', 'error', 'bug', 'fix', 'update', 'version', 'config']
    
    texts = []
    labels = []
    
    # Create performance bugs
    for i in range(n_perf):
        words = np.random.choice(perf_keywords, np.random.randint(3, 8))
        text = ' '.join(words)
        labels.append(1)
        texts.append(text)
    
    # Create non-performance bugs
    for i in range(n_non_perf):
        words = np.random.choice(other_keywords, np.random.randint(3, 8))
        text = ' '.join(words)
        labels.append(0)
        texts.append(text)
    
    # Shuffle
    indices = np.random.permutation(len(texts))
    texts = [texts[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    print(f"Created synthetic {n_samples} samples with {n_perf} performance bugs ({perf_ratio*100:.1f}%)")
    return np.array(texts), np.array(labels)

def evaluate_predictions(y_true, y_pred):
    """Calculate classification metrics"""
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

def save_results(results, filename='experiment_results.csv'):
    """Save results to CSV"""
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    df = pd.DataFrame(results)
    df.to_csv(f'data/processed/{filename}', index=False)
    print(f"Results saved to data/processed/{filename}")

def plot_comparison(baseline_scores, your_scores, metric='F1 Score'):
    """Create comparison plot"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    # Create box plot
    data_to_plot = [baseline_scores, your_scores]
    bp = plt.boxplot(data_to_plot, patch_artist=True)
    
    # Customize colors
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    plt.xticks([1, 2], ['Baseline (Naive Bayes)', 'FairBug (Our Solution)'])
    plt.ylabel(metric)
    plt.title(f'Comparison of {metric} across Experiments')
    plt.grid(True, alpha=0.3)
    
    # Add individual points
    plt.scatter(np.random.normal(1, 0.04, len(baseline_scores)), baseline_scores, 
               alpha=0.5, color='blue', label='Baseline')
    plt.scatter(np.random.normal(2, 0.04, len(your_scores)), your_scores, 
               alpha=0.5, color='green', label='FairBug')
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    os.makedirs('data/processed', exist_ok=True)
    plt.savefig(f'data/processed/comparison_{metric.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()