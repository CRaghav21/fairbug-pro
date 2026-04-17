"""
Fairness Testing Module for Bug Report Classifier
Author: Your Name
Date: 2024
"""

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FairnessTester:
    """
    Test fairness of bug report classifier across different projects
    and identify discriminatory instances
    """
    
    def __init__(self, classifier):
        """
        Initialize fairness tester
        
        Args:
            classifier: Trained classifier (EnhancedBugClassifier)
        """
        self.classifier = classifier
        
    def test_project_fairness(self, data_a, data_b, project_a_name, project_b_name):
        """
        Test if classifier treats two projects fairly
        """
        texts_a, labels_a = data_a
        texts_b, labels_b = data_b
        
        print(f"Testing fairness between {project_a_name} and {project_b_name}...")
        print(f"  {project_a_name}: {len(texts_a)} samples, {sum(labels_a)} performance bugs")
        print(f"  {project_b_name}: {len(texts_b)} samples, {sum(labels_b)} performance bugs")
        
        # Getting predictions
        preds_a = self.classifier.predict_ensemble(texts_a)
        preds_b = self.classifier.predict_ensemble(texts_b)
        
        # Calculating error rates
        error_a = np.mean(preds_a != labels_a)
        error_b = np.mean(preds_b != labels_b)
        
        # Calculating F1 scores
        from sklearn.metrics import f1_score
        f1_a = f1_score(labels_a, preds_a, zero_division=0)
        f1_b = f1_score(labels_b, preds_b, zero_division=0)
        
        # Statistical test
        correct_a = np.sum(preds_a == labels_a)
        incorrect_a = len(labels_a) - correct_a
        correct_b = np.sum(preds_b == labels_b)
        incorrect_b = len(labels_b) - correct_b
        
        contingency_table = np.array([
            [correct_a, incorrect_a],
            [correct_b, incorrect_b]
        ])
        
        if np.min(contingency_table) == 0:
            contingency_table = contingency_table + 0.001
        
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        except:
            p_value = 1.0
        
        fairness_gap = abs(error_a - error_b)
        pos_rate_a = np.mean(preds_a == 1) if len(preds_a) > 0 else 0
        pos_rate_b = np.mean(preds_b == 1) if len(preds_b) > 0 else 0
        
        if max(pos_rate_a, pos_rate_b) > 0:
            disparate_impact = min(pos_rate_a, pos_rate_b) / max(pos_rate_a, pos_rate_b)
        else:
            disparate_impact = 1.0
        
        return {
            'project_a': project_a_name,
            'project_b': project_b_name,
            'error_rate_a': error_a,
            'error_rate_b': error_b,
            'f1_score_a': f1_a,
            'f1_score_b': f1_b,
            'fairness_gap': fairness_gap,
            'p_value': p_value,
            'is_fair': p_value > 0.05 if not np.isnan(p_value) else True,
            'disparate_impact': disparate_impact,
            'pos_rate_a': pos_rate_a,
            'pos_rate_b': pos_rate_b,
            'samples_a': len(texts_a),
            'samples_b': len(texts_b),
            'perf_bugs_a': int(sum(labels_a)),
            'perf_bugs_b': int(sum(labels_b))
        }
    
    def generate_discriminatory_pairs(self, texts, labels, n_pairs=50):
        """
        Generate pairs of similar reports that get different predictions
        OPTIMIZED VERSION - Much faster!
        """
        print(f"\nGenerating {n_pairs} discriminatory pairs...")
        
        # Getting predictions first (this is the slow part)
        print("Making predictions on all samples...")
        preds = self.classifier.predict_ensemble(texts)
        
        # Finding indices where predictions differ from labels
        misclassified = np.where(preds != labels)[0]
        
        if len(misclassified) < 2:
            print(f"⚠ Only {len(misclassified)} misclassified samples found. Cannot generate pairs.")
            return []
        
        print(f"Found {len(misclassified)} misclassified samples")
        
        # Limiting to first 500 for speed
        if len(misclassified) > 500:
            misclassified = misclassified[:500]
            print(f"Using first 500 misclassified samples for efficiency")
        
        # Calculating text similarities only for misclassified samples
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        print("Calculating text similarities...")
        misclassified_texts = [texts[i] for i in misclassified]
        
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(misclassified_texts)
        
        # Finding discriminatory pairs
        discriminatory_pairs = []
        n_samples = len(misclassified)
        
        print(f"Searching for pairs among {n_samples} samples...")
        
        # Using progress bar
        from tqdm import tqdm
        pbar = tqdm(total=min(n_pairs, n_samples * n_samples), desc="Finding pairs")
        
        attempts = 0
        max_attempts = n_pairs * 10  # Limiting attempts to avoid infinite loop
        
        while len(discriminatory_pairs) < n_pairs and attempts < max_attempts:
            # Randomly select two different misclassified samples
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            
            # Calculating similarity
            similarity = cosine_similarity(tfidf_matrix[idx1:idx1+1], tfidf_matrix[idx2:idx2+1])[0][0]
            
            # Checking if they're similar enough
            if similarity > 0.3:
                original_idx1 = misclassified[idx1]
                original_idx2 = misclassified[idx2]
                
                discriminatory_pairs.append({
                    'report_a': texts[original_idx1][:300] + '...' if len(texts[original_idx1]) > 300 else texts[original_idx1],
                    'report_b': texts[original_idx2][:300] + '...' if len(texts[original_idx2]) > 300 else texts[original_idx2],
                    'label_a': int(labels[original_idx1]),
                    'label_b': int(labels[original_idx2]),
                    'pred_a': int(preds[original_idx1]),
                    'pred_b': int(preds[original_idx2]),
                    'similarity': similarity
                })
                pbar.update(1)
            
            attempts += 1
        
        pbar.close()
        
        print(f"\n✓ Found {len(discriminatory_pairs)} discriminatory pairs")
        
        # Saving sample pairs to CSV
        if len(discriminatory_pairs) > 0:
            pairs_df = pd.DataFrame(discriminatory_pairs)
            pairs_df.to_csv('data/processed/discriminatory_pairs.csv', index=False)
            print(f"✓ Saved to data/processed/discriminatory_pairs.csv")
            
            # Showing first few pairs
            print("\nSample discriminatory pairs:")
            for i, pair in enumerate(discriminatory_pairs[:3]):
                print(f"\nPair {i+1}:")
                print(f"  Report A: {pair['report_a'][:100]}...")
                print(f"  Report B: {pair['report_b'][:100]}...")
                print(f"  Similarity: {pair['similarity']:.3f}")
                print(f"  Labels: {pair['label_a']} vs {pair['label_b']}")
                print(f"  Predictions: {pair['pred_a']} vs {pair['pred_b']}")
        
        return discriminatory_pairs
    
    def calculate_fairness_metrics_across_projects(self, projects_data):
        """
        Calculate fairness metrics for all pairs of projects
        """
        project_names = list(projects_data.keys())
        results = []
        
        print("\nTesting fairness across projects...")
        
        for i in range(len(project_names)):
            for j in range(i+1, len(project_names)):
                proj_a = project_names[i]
                proj_b = project_names[j]
                
                fairness_result = self.test_project_fairness(
                    projects_data[proj_a],
                    projects_data[proj_b],
                    proj_a,
                    proj_b
                )
                results.append(fairness_result)
        
        return pd.DataFrame(results)
    
    def statistical_significance_test(self, baseline_errors, your_errors):
        """
        Perform statistical significance test
        """
        t_stat, p_value_t = stats.ttest_rel(baseline_errors, your_errors)
        w_stat, p_value_w = stats.wilcoxon(baseline_errors, your_errors)
        
        return {
            't_statistic': t_stat,
            't_test_p_value': p_value_t,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': p_value_w,
            'significantly_better': p_value_w < 0.05,
            'mean_improvement': np.mean(baseline_errors) - np.mean(your_errors),
            'median_improvement': np.median(baseline_errors) - np.median(your_errors)
        }