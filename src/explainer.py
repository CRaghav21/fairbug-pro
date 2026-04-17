"""
Explainability Module for Bug Report Classifier
Author: Raghavendra J Chigarahalli
Date: 5th Apr 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BugReportExplainer:
    """
    Explain predictions of bug report classifier
    """
    
    def __init__(self, classifier):
        """
        Initialize explainer
        
        Args:
            classifier: Trained EnhancedBugClassifier
        """
        self.classifier = classifier
        self.vectorizer = classifier.vectorizer
        
        # Featuring names from vectorizer
        self.feature_names = None
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()
    
    def explain_prediction(self, report_text):
        """
        Explain why a report was classified as performance bug or not
        
        Args:
            report_text (str): Bug report text
        
        Returns:
            dict: Explanation of prediction
        """
        # Getting prediction
        pred = self.classifier.predict_single(report_text)
        proba = self.classifier.predict_proba_ensemble([report_text])[0]
        
        # Convertting to scalar if needed
        if isinstance(pred, np.ndarray):
            pred = pred[0] if len(pred) > 0 else 0
        
        # Getting confidence as scalar
        confidence = float(proba[pred]) if pred < len(proba) else 0.5
        
        # Getting feature importance for this prediction
        X = self.vectorizer.transform([report_text])
        
        # Getting non-zero features in this report
        non_zero_indices = X.nonzero()[1]
        
        # Extracting important words
        word_importance = []
        
        if hasattr(self.classifier.classifiers['random_forest'], 'feature_importances_'):
            importances = self.classifier.classifiers['random_forest'].feature_importances_
            
            for idx in non_zero_indices:
                if idx < len(importances) and idx < len(self.feature_names):
                    word = self.feature_names[idx]
                    importance = float(importances[idx])
                    word_importance.append((word, importance))
        
        # Sorting by importance
        word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generating human-readable explanation
        explanation = self._generate_explanation(word_importance[:10], pred, confidence)
        
        return {
            'prediction': 'Performance Bug' if pred == 1 else 'Not Performance Bug',
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'top_words': word_importance[:10],
            'explanation': explanation,
            'original_text': report_text[:200] + '...' if len(report_text) > 200 else report_text
        }
    
    def _generate_explanation(self, top_words, prediction, confidence):
        """
        Generate human-readable explanation
        
        Args:
            top_words (list): List of (word, importance) tuples
            prediction (int): 0 or 1
            confidence (float): Prediction confidence
        
        Returns:
            str: Human-readable explanation
        """
        if len(top_words) == 0:
            return "Unable to generate detailed explanation. Report may be too short or contain only stopwords."
        
        words = [word for word, _ in top_words[:5]]
        
        # Making sure confidence is a float
        conf_float = float(confidence)
        
        if prediction == 1:
            explanation = f"This report is classified as a performance bug with {conf_float*100:.1f}% confidence.\n"
            explanation += f"Key performance-related terms detected: {', '.join(words[:5])}.\n"
            explanation += "These terms are strongly associated with performance issues in our training data."
        else:
            explanation = f"This report is NOT classified as a performance bug with {conf_float*100:.1f}% confidence.\n"
            explanation += f"Key terms detected: {', '.join(words[:5])}.\n"
            explanation += "These terms are not strongly associated with performance issues in our training data."
        
        return explanation
    
    def batch_explain(self, reports, output_file='sample_explanations.csv'):
        """
        Generate explanations for multiple reports
        
        Args:
            reports (list): List of bug reports
            output_file (str): Output CSV file path
        
        Returns:
            pd.DataFrame: Explanations for all reports
        """
        explanations = []
        
        for i, report in enumerate(reports):
            if i % 10 == 0:
                print(f"Explaining report {i}/{len(reports)}")
            
            try:
                exp = self.explain_prediction(report)
                explanations.append({
                    'report_index': i,
                    'text_preview': report[:100],
                    'prediction': exp['prediction'],
                    'confidence': exp['confidence'],
                    'explanation': exp['explanation'],
                    'top_words': '; '.join([f"{w}({i:.3f})" for w, i in exp['top_words'][:5]])
                })
            except Exception as e:
                print(f"Error explaining report {i}: {e}")
                continue
        
        df = pd.DataFrame(explanations)
        df.to_csv(f'data/processed/{output_file}', index=False)
        print(f"✓ Explanations saved to data/processed/{output_file}")
        
        return df
    
    def visualize_feature_importance(self, top_n=20):
        """
        Visualize most important features globally
        
        Args:
            top_n (int): Number of top features to show
        
        Returns:
            pd.DataFrame or None: Feature importance data
        """
        if not hasattr(self.classifier.classifiers['random_forest'], 'feature_importances_'):
            print("Feature importances not available")
            return None
        
        importances = self.classifier.classifiers['random_forest'].feature_importances_
        
        # Geting top N features
        top_indices = np.argsort(importances)[-top_n:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        # Creating plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances, color='steelblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features for Performance Bug Detection')
        plt.tight_layout()
        
        plt.savefig('data/processed/feature_importance.png', dpi=300)
        print("✓ Feature importance plot saved to data/processed/feature_importance.png")
        plt.show()
        
        # Returning as DataFrame
        importance_df = pd.DataFrame({
            'feature': top_features,
            'importance': top_importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
