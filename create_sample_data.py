"""
Create sample bug report data for testing
"""

import pandas as pd
import numpy as np
import os

# Create data directory
os.makedirs('data/raw', exist_ok=True)

# Sample data for each project
projects_data = {
    'tensorflow': {
        'n_samples': 1490,
        'perf_ratio': 0.187,  # 18.7% performance bugs
        'perf_keywords': ['slow', 'memory', 'gpu', 'performance', 'crash', 'timeout', 'speed', 'latency'],
        'other_keywords': ['feature', 'doc', 'install', 'error', 'bug', 'fix', 'update', 'api']
    },
    'pytorch': {
        'n_samples': 752,
        'perf_ratio': 0.126,
        'perf_keywords': ['slow', 'memory', 'cuda', 'performance', 'crash', 'timeout', 'speed', 'gpu'],
        'other_keywords': ['feature', 'doc', 'install', 'error', 'bug', 'fix', 'update', 'tensor']
    },
    'keras': {
        'n_samples': 668,
        'perf_ratio': 0.202,
        'perf_keywords': ['slow', 'memory', 'performance', 'crash', 'timeout', 'speed', 'training'],
        'other_keywords': ['feature', 'doc', 'install', 'error', 'bug', 'fix', 'layer', 'model']
    },
    'mxnet': {
        'n_samples': 516,
        'perf_ratio': 0.126,
        'perf_keywords': ['slow', 'memory', 'performance', 'crash', 'timeout', 'gpu', 'speed'],
        'other_keywords': ['feature', 'doc', 'install', 'error', 'bug', 'fix', 'symbol', 'ndarray']
    },
    'caffe': {
        'n_samples': 286,
        'perf_ratio': 0.115,
        'perf_keywords': ['slow', 'memory', 'performance', 'crash', 'gpu', 'speed', 'cuda'],
        'other_keywords': ['feature', 'doc', 'install', 'error', 'bug', 'fix', 'layer', 'blob']
    }
}

def create_sample_data(project_name, config):
    """Create sample bug report data"""
    np.random.seed(42)
    
    data = []
    n_perf = int(config['n_samples'] * config['perf_ratio'])
    n_non_perf = config['n_samples'] - n_perf
    
    # Create performance bug reports
    for i in range(n_perf):
        words = np.random.choice(config['perf_keywords'], np.random.randint(3, 10))
        title = ' '.join(words)
        description = f"Performance issue: {' '.join(words)}. The system becomes very slow when running."
        data.append({
            'report_id': f"{project_name}_perf_{i}",
            'title': title,
            'description': description,
            'label': 1,
            'created_at': f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}",
            'status': np.random.choice(['open', 'closed'], p=[0.3, 0.7]),
            'comments_count': np.random.randint(0, 15)
        })
    
    # Create non-performance bug reports
    for i in range(n_non_perf):
        words = np.random.choice(config['other_keywords'], np.random.randint(3, 10))
        title = ' '.join(words)
        description = f"Feature request: {' '.join(words)}. Please add this functionality."
        data.append({
            'report_id': f"{project_name}_nonperf_{i}",
            'title': title,
            'description': description,
            'label': 0,
            'created_at': f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,28):02d}",
            'status': np.random.choice(['open', 'closed'], p=[0.3, 0.7]),
            'comments_count': np.random.randint(0, 15)
        })
    
    # Shuffle data
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    filename = f'data/raw/{project_name}_reports.csv'
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {len(df)} samples ({n_perf} performance bugs)")

# Create all datasets
for project, config in projects_data.items():
    create_sample_data(project, config)

print("\n✓ All sample datasets created successfully!")
print("\nNow run: python main.py")