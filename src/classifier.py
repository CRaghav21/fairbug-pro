"""
Enhanced Bug Report Classifier with Ensemble Methods
Author: Raghavendra J Chigarahalli
Date: 5th Apr 2026
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedBugClassifier:
    """
    Ensemble classifier for bug report classification
    Uses multiple ML models and combines their predictions
    """
    
    def __init__(self, max_features=5000, ngram_range=(1, 3)):
        """
        Initialize the classifier
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            ngram_range (tuple): Range of n-grams to use
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initializing TF-IDF vectorizer with advanced parameters
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,  # Use sublinear scaling
            use_idf=True
        )
        
        # Initializing individual classifiers
        self.classifiers = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='auto',
                probability=True,
                random_state=42
            )
        }
        
        self.is_trained = False
        
    def train_ensemble(self, X_train, y_train):
        """
        Train all classifiers in the ensemble
        
        Args:
            X_train (list): Training texts
            y_train (array): Training labels
        """
        print("Training ensemble classifier...")
        
        # Transforming texts to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"TF-IDF shape: {X_train_tfidf.shape}")
        
        # Training each classifier
        for name, clf in self.classifiers.items():
            print(f"Training {name}...")
            clf.fit(X_train_tfidf, y_train)
        
        self.is_trained = True
        print("Training completed!")
    
    def predict_ensemble(self, X):
        """
        Predict using ensemble voting
        
        Args:
            X (list): Texts to predict
        
        Returns:
            array: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train_ensemble first.")
        
        X_tfidf = self.vectorizer.transform(X)
        predictions = []
        
        for name, clf in self.classifiers.items():
            pred = clf.predict(X_tfidf)
            predictions.append(pred)
        
        # Stacking predictions and use majority voting
        predictions = np.array(predictions)
        final_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=predictions
        )
        
        return final_pred
    
    def predict_proba_ensemble(self, X):
        """
        Get probability predictions from ensemble
        
        Args:
            X (list): Texts to predict
        
        Returns:
            array: Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train_ensemble first.")
        
        X_tfidf = self.vectorizer.transform(X)
        probabilities = []
        
        for name, clf in self.classifiers.items():
            if hasattr(clf, 'predict_proba'):
                proba = clf.predict_proba(X_tfidf)
            else:
                # For SVM without probability, use decision function
                proba = clf.decision_function(X_tfidf)
                proba = np.column_stack([-proba, proba])
                proba = np.exp(proba) / np.exp(proba).sum(axis=1, keepdims=True)
            probabilities.append(proba)
        
        # Averaging probabilities
        return np.mean(probabilities, axis=0)
    
    def predict_single(self, text):
        """
        Predict single bug report
        
        Args:
            text (str): Bug report text
        
        Returns:
            int: Prediction (0 or 1)
        """
        return self.predict_ensemble([text])[0]
    
    def save_model(self, filepath='models/fairbug_model.joblib'):
        """
        Save trained model to disk
        
        Args:
            filepath (str): Path to save model
        """
        import os
        
        # Creating directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'vectorizer': self.vectorizer,
            'classifiers': self.classifiers,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/fairbug_model.joblib'):
        """
        Load trained model from disk
        
        Args:
            filepath (str): Path to load model
        """
        model_data = joblib.load(filepath)
        self.vectorizer = model_data['vectorizer']
        self.classifiers = model_data['classifiers']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

# Baseline classifier (Naive Bayes) for comparison
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class BaselineClassifier:
    """
    Baseline classifier using Naive Bayes + TF-IDF
    """
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )),
            ('nb', MultinomialNB())
        ])
    
    def train(self, X_train, y_train):
        """Train the baseline model"""
        self.pipeline.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions"""
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        return self.pipeline.predict_proba(X)
