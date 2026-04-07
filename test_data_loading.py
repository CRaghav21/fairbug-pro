"""
Test script to verify data loading
"""

from src.utils import load_dataset

# Test loading each dataset
projects = ['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe']

for project in projects:
    print(f"\n{'='*40}")
    print(f"Testing {project}")
    print(f"{'='*40}")
    
    texts, labels = load_dataset(project)
    
    print(f"✓ Loaded {len(texts)} samples")
    print(f"✓ Performance bugs: {sum(labels)}")
    print(f"✓ Sample text: {texts[0][:100]}...")
    print(f"✓ Sample label: {labels[0]}")