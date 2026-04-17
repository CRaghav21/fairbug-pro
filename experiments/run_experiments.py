"""
Main experiment runner for FairBug
Author: Raghavendra J Chigarahalli
Date: April 2nd 2026
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.classifier import EnhancedBugClassifier, BaselineClassifier
from src.fairness_tester import FairnessTester
from src.explainer import BugReportExplainer
from src.utils import load_dataset, evaluate_predictions, save_results, plot_comparison
from src.time_analyzer import TimeAwareAnalyzer
from src.severity_predictor import SeverityPredictor
from src.cross_language import CrossLanguageDetector

class ExperimentRunner:

    
    def __init__(self):
        self.results = {
            'project': [],
            'experiment_id': [],
            'model': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        self.baseline_scores = []
        self.fairbug_scores = []
        
        # Created processed data directory if it doesn't existists
        os.makedirs('data/processed', exist_ok=True)
    
    def run_classification_experiments(self, n_repeats=30, test_size=0.3):
        
        print("=" * 60)
        print("Running Classification Experiments")
        print("=" * 60)
        
        projects = ['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe']
        
        for project in projects:
            print(f"\n{'='*40}")
            print(f"Processing project: {project}")
            print(f"{'='*40}")
            
            # Loading data for this project
            texts, labels = load_dataset(project)
            
            if len(texts) == 0:
                print(f"Warning: No data for {project}, skipping...")
                continue
            
            print(f"Total samples: {len(texts)}")
            print(f"Performance bugs: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
            
            # Running repeated experiments
            for repeat in range(n_repeats):
                if repeat % 10 == 0:
                    print(f"  Repeat {repeat+1}/{n_repeats}")
                
                # Spliting data
                X_train, X_test, y_train, y_test = train_test_split(
                    texts, labels, test_size=test_size, 
                    random_state=repeat, stratify=labels
                )
                
                # Baseline: Naive Bayes
                baseline = BaselineClassifier()
                baseline.train(X_train, y_train)
                y_pred_baseline = baseline.predict(X_test)
                
                # My solution: FairBug
                fairbug = EnhancedBugClassifier()
                fairbug.train_ensemble(X_train, y_train)
                y_pred_fairbug = fairbug.predict_ensemble(X_test)
                
                # Calculating metrics
                metrics_baseline = evaluate_predictions(y_test, y_pred_baseline)
                metrics_fairbug = evaluate_predictions(y_test, y_pred_fairbug)
                
                # Storing results
                for metric, value in metrics_baseline.items():
                    self.results[metric].append(value)
                self.results['project'].append(project)
                self.results['experiment_id'].append(repeat)
                self.results['model'].append('baseline')
                
                for metric, value in metrics_fairbug.items():
                    self.results[metric].append(value)
                self.results['project'].append(project)
                self.results['experiment_id'].append(repeat)
                self.results['model'].append('fairbug')
                
                # Storing F1 scores for statistical test
                self.baseline_scores.append(metrics_baseline['f1_score'])
                self.fairbug_scores.append(metrics_fairbug['f1_score'])
        
        # Converting results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Saving results to data/processed paths
        output_path = 'data/processed/classification_results.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Classification results saved to: {output_path}")
        
        return results_df
    
    def run_statistical_tests(self):
        
        print("\n" + "=" * 60)
        print("Statistical Significance Tests")
        print("=" * 60)
        
        # Converting into arrays
        baseline_scores = np.array(self.baseline_scores)
        fairbug_scores = np.array(self.fairbug_scores)
        
        # Calculating mean improvements
        mean_improvement = np.mean(fairbug_scores - baseline_scores)
        median_improvement = np.median(fairbug_scores - baseline_scores)
        
        print(f"\nPerformance Summary:")
        print(f"Baseline Mean F1: {np.mean(baseline_scores):.4f} ± {np.std(baseline_scores):.4f}")
        print(f"FairBug Mean F1: {np.mean(fairbug_scores):.4f} ± {np.std(fairbug_scores):.4f}")
        print(f"Mean Improvement: {mean_improvement:.4f}")
        print(f"Median Improvement: {median_improvement:.4f}")
        
        # Paired t-test
        t_stat, p_value_t = stats.ttest_rel(baseline_scores, fairbug_scores)
        
        # Wilcoxon signed-rank test
        w_stat, p_value_w = stats.wilcoxon(baseline_scores, fairbug_scores)
        
        print(f"\nStatistical Test Results:")
        print(f"Paired t-test: t = {t_stat:.4f}, p = {p_value_t:.4f}")
        print(f"Wilcoxon test: W = {w_stat:.4f}, p = {p_value_w:.4f}")
        
        if p_value_w < 0.05:
            print("\n✓ FairBug is significantly better than baseline (p < 0.05)")
        else:
            print("\n✗ No significant difference detected (p ≥ 0.05)")
        
        # Createing comparison plots and saving into data/processed paths
        self.plot_comparison(baseline_scores, fairbug_scores, 'F1 Score')
        
        # Saving statistical results
        stats_results = {
            't_statistic': t_stat,
            't_p_value': p_value_t,
            'wilcoxon_statistic': w_stat,
            'wilcoxon_p_value': p_value_w,
            'mean_improvement': mean_improvement,
            'baseline_mean': np.mean(baseline_scores),
            'fairbug_mean': np.mean(fairbug_scores)
        }
        
        stats_df = pd.DataFrame([stats_results])
        stats_df.to_csv('data/processed/statistical_tests.csv', index=False)
        print(f"\n✓ Statistical test results saved to: data/processed/statistical_tests.csv")
        
        return stats_results
    
    def plot_comparison(self, baseline_scores, your_scores, metric='F1 Score'):
        
        plt.figure(figsize=(10, 6))
        
        # Creating box plot
        data_to_plot = [baseline_scores, your_scores]
        bp = plt.boxplot(data_to_plot, patch_artist=True)
        
        # Customizing colors
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightgreen')
        
        plt.xticks([1, 2], ['Baseline (Naive Bayes)', 'FairBug (Our Solution)'])
        plt.ylabel(metric)
        plt.title(f'Comparison of {metric} across 30 Experiments')
        plt.grid(True, alpha=0.3)
        
        # Adding individual points
        plt.scatter(np.random.normal(1, 0.04, len(baseline_scores)), baseline_scores, 
                   alpha=0.5, color='blue', label='Baseline')
        plt.scatter(np.random.normal(2, 0.04, len(your_scores)), your_scores, 
                   alpha=0.5, color='green', label='FairBug')
        
        plt.legend()
        plt.tight_layout()
        
        # Saving plots
        plt.savefig('data/processed/comparison_f1_score.png', dpi=300)
        print(f"✓ Comparison plot saved to: data/processed/comparison_f1_score.png")
        plt.show()
    
    def run_fairness_experiments(self):
        
        print("\n" + "=" * 60)
        print("Running Fairness Experiments")
        print("=" * 60)
        
        # Loading data for all projects
        projects = ['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe']
        projects_data = {}
        
        for project in projects:
            texts, labels = load_dataset(project)
            if len(texts) > 0:
                projects_data[project] = (texts, labels)
                print(f"Loaded {project}: {len(texts)} samples")
        
        # Training FairBug on combined data
        print("\nTraining FairBug on combined dataset...")
        combined_texts = []
        combined_labels = []
        
        for project, (texts, labels) in projects_data.items():
            combined_texts.extend(texts)
            combined_labels.extend(labels)
        
        fairbug = EnhancedBugClassifier()
        fairbug.train_ensemble(combined_texts, combined_labels)
        
        # Testing fairness
        fairness_tester = FairnessTester(fairbug)
        fairness_results = fairness_tester.calculate_fairness_metrics_across_projects(projects_data)
        
        print("\nFairness Test Results:")
        print(fairness_results[['project_a', 'project_b', 'fairness_gap', 'p_value', 'is_fair']])
        
        # Saving fairness results to data/processed
        fairness_results.to_csv('data/processed/fairness_results.csv', index=False)
        print(f"\n✓ Fairness results saved to: data/processed/fairness_results.csv")
        
        # Generating discriminatory pairs for a sample project
        print("\nGenerating discriminatory pairs for TensorFlow...")
        if 'tensorflow' in projects_data:
            texts, labels = projects_data['tensorflow']
            discriminatory_pairs = fairness_tester.generate_discriminatory_pairs(texts, labels, n_pairs=100)
            
            # Saving pairs
            pairs_df = pd.DataFrame(discriminatory_pairs)
            pairs_df.to_csv('data/processed/discriminatory_pairs.csv', index=False)
            print(f"✓ Found {len(discriminatory_pairs)} discriminatory pairs")
            print(f"✓ Saved to: data/processed/discriminatory_pairs.csv")
        
        return fairness_results
    
    def run_explainability_demo(self):
    
        print("\n" + "=" * 60)
        print("Explainability Demo")
        print("=" * 60)
        
        # Loading sample data
        texts, labels = load_dataset('tensorflow')
        
        # Training classifier
        fairbug = EnhancedBugClassifier()
        fairbug.train_ensemble(texts[:300], labels[:300])  # Use subset for speed
        
        # Initializing explainer
        explainer = BugReportExplainer(fairbug)
        
        # Getting some sample reports
        sample_reports = texts[:5]
        
        print("\nAnalyzing sample bug reports:")
        print("-" * 40)
        
        explanations_list = []
        
        for i, report in enumerate(sample_reports):
            print(f"\nReport {i+1}:")
            print(f"Text: {report[:200]}...")
            
            explanation = explainer.explain_prediction(report)
            print(f"Prediction: {explanation['prediction']}")
            print(f"Confidence: {explanation['confidence_percentage']}")
            print(f"Explanation: {explanation['explanation']}")
            print(f"Top words: {[w for w, _ in explanation['top_words'][:5]]}")
            
            explanations_list.append({
                'report_id': i+1,
                'text_preview': report[:200],
                'prediction': explanation['prediction'],
                'confidence': explanation['confidence'],
                'explanation': explanation['explanation'],
                'top_words': ', '.join([w for w, _ in explanation['top_words'][:5]])
            })
        
        # Saving explanations to data/processed
        explanations_df = pd.DataFrame(explanations_list)
        explanations_df.to_csv('data/processed/sample_explanations.csv', index=False)
        print(f"\n✓ Sample explanations saved to: data/processed/sample_explanations.csv")
        
        # Visualizing feature importance and save
        print("\nVisualizing global feature importance...")
        importance_df = explainer.visualize_feature_importance(top_n=20)
        if importance_df is not None:
            importance_df.to_csv('data/processed/feature_importance.csv', index=False)
            print(f"✓ Feature importance saved to: data/processed/feature_importance.csv")
        
        return explainer
    def run_novel_analysis(self):
      print("\n" + "=" * 60)
      print("NOVEL ANALYSIS: Time, Severity & Cross-Language")
      print("=" * 60)
    
      # Loading data for analysis
      project = 'tensorflow'  # Use largest dataset
      texts, labels = load_dataset(project)
    
      # Getting dates from original data
      import pandas as pd
      filepath = f'data/raw/{project}_reports.csv'
      df = pd.read_csv(filepath)
      dates = df['created_at'].values if 'created_at' in df.columns else [None] * len(texts)
    
      # 1. TIME-AWARE ANALYSIS
      print("\n📊 1. TIME-AWARE BUG ANALYSIS")
      print("-" * 40)
      time_analyzer = TimeAwareAnalyzer()
      trend_results = time_analyzer.analyze_bug_trends(texts, labels, dates)
    
      print(f"   Total performance bugs: {trend_results['total_performance_bugs']}")
      print(f"   Performance bug rate: {trend_results['performance_rate']:.1%}")
      print(f"   Trend: {trend_results['trend']}")
      print(f"   Busiest month for performance bugs: Month {trend_results['busiest_month']}")
    
      # Saving time analysis results
      import json
      with open('data/processed/time_analysis.json', 'w') as f:
          # Converting non-serializable objects
          serializable_results = {
              'peak_performance_month': trend_results['peak_performance_month'],
              'trend': trend_results['trend'],
              'busiest_month': trend_results['busiest_month'],
              'total_performance_bugs': trend_results['total_performance_bugs'],
              'performance_rate': float(trend_results['performance_rate']),
              'seasonal_pattern': {str(k): float(v) for k, v in trend_results['seasonal_pattern'].items()}
          }
          json.dump(serializable_results, f, indent=2)
      print("   ✓ Time analysis saved to data/processed/time_analysis.json")
    
      # Creating plot
      time_analyzer.plot_temporal_analysis(trend_results, project)
    
      # 2. SEVERITY PREDICTION
      print("\n⚠️ 2. BUG SEVERITY PREDICTION")
      print("-" * 40)
      severity_predictor = SeverityPredictor()
    
      # Analysing first 10 reports
      sample_reports = texts[:10]
      sample_dates = dates[:10]
    
      severity_results = []
      for i, (report, date) in enumerate(zip(sample_reports, sample_dates)):
          severity = severity_predictor.predict_severity(report, comments_count=np.random.randint(0, 15))
          severity_results.append(severity)
        
          print(f"\n   Report {i+1}:")
          print(f"   Text: {report[:80]}...")
          print(f"   Severity: {severity['severity_score']}/5 - {severity['severity_level']}")
          print(f"   Fix Time: {severity['estimated_fix_time']}")
          print(f"   Action: {severity['recommended_action']}")
    
      # Saving severity results
      severity_df = pd.DataFrame(severity_results)
      severity_df.to_csv('data/processed/severity_analysis.csv', index=False)
      print("\n   ✓ Severity analysis saved to data/processed/severity_analysis.csv")
    
      # 3. CROSS-LANGUAGE DETECTION
      print("\n🌐 3. CROSS-LANGUAGE PERFORMANCE BUG DETECTION")
      print("-" * 40)
      cross_lang = CrossLanguageDetector()
    
      # Testing with different language pairs
      languages = ['python', 'c_cpp', 'java', 'javascript']
    
      # Finding a performance bug report
      perf_indices = [i for i, label in enumerate(labels) if label == 1]
      if perf_indices:
          sample_perf_bug = texts[perf_indices[0]]
        
          print(f"\n   Sample Performance Bug:")
          print(f"   {sample_perf_bug[:150]}...")
        
          cross_lang_results = []
          for target_lang in languages[1:]:  # Skip source language
              result = cross_lang.detect_cross_language_pattern(
                  sample_perf_bug, 
                  source_lang='python', 
                  target_lang=target_lang
              )
              cross_lang_results.append(result)
              
              print(f"\n   Python → {target_lang.upper()}:")
              print(f"   Matched patterns: {result['matched_patterns_in_source']}")
              print(f"   Cross-language score: {result['cross_language_score']:.2f}")
              print(f"   {result['recommendation']}")
        
          # Saving cross-language results
          cross_df = pd.DataFrame(cross_lang_results)
          cross_df.to_csv('data/processed/cross_language_analysis.csv', index=False)
          print("\n   ✓ Cross-language analysis saved to data/processed/cross_language_analysis.csv")
      
      print("\n" + "=" * 60)
      print("✓ NOVEL ANALYSIS COMPLETE!")
      print("=" * 60)
    
def main():
    print("=" * 60)
    print("FAIRBUG PRO: Advanced Bug Report Intelligence System")
    print("=" * 60)
    print("\nStarting experiments...\n")
    
    # Creating results directory
    os.makedirs('data/processed', exist_ok=True)
    
    # Initializing experiment runner
    runner = ExperimentRunner()
    
    # Running classification experiments
    classification_results = runner.run_classification_experiments(n_repeats=30)
    
    # Printing summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS SUMMARY")
    print("=" * 60)
    
    summary = classification_results.groupby('model').agg({
        'f1_score': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'accuracy': ['mean', 'std']
    }).round(4)
    
    print(summary)
    
    # Running statistical tests
    stat_results = runner.run_statistical_tests()
    
    # Running fairness experiments
    fairness_results = runner.run_fairness_experiments()
    
    # Running explainability demo
    explainer = runner.run_explainability_demo()
    
    # ★★★ NEW: Run novel analysis ★★★
    runner.l_analysis()
    
    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nResults saved to: data/processed/")
    print("\nFiles generated:")
    print("  - classification_results.csv")
    print("  - statistical_tests.csv")
    print("  - comparison_f1_score.png")
    print("  - fairness_results.csv")
    print("  - discriminatory_pairs.csv")
    print("  - sample_explanations.csv")
    print("  - feature_importance.csv")
    print("  ★ time_analysis.json")
    print("  ★ temporal_analysis_tensorflow.png")
    print("  ★ severity_analysis.csv")
    print("  ★ cross_language_analysis.csv")
    print("\nKey findings:")
    print(f"- FairBug improves F1 score by {stat_results['mean_improvement']:.4f} on average")
    print(f"- Statistical significance: p = {stat_results['wilcoxon_p_value']:.4f}")
    print(f"- Found fairness gaps across projects")
    print(f"- Generated explainable predictions for bug reports")
    print(f"- ★ Novel: Temporal patterns, severity prediction, cross-language detection")

if __name__ == "__main__":
    main()