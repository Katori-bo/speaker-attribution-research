import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PointwiseLogisticRanker:
    """
    A Pointwise Candidate Ranker using Logistic Regression.
    For a given quote, it scores all candidates and picks the one with the highest probability.
    """
    def __init__(self, random_state=42, max_iter=1000):
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter, class_weight='balanced')
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the logistic regression model on the feature matrix X.
        """
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns the probability of class 1 for each row.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
        
    def evaluate_ranking(self, df: pd.DataFrame) -> dict:
        """
        Evaluates the ranking accuracy on a dataframe grouped by quote_id.
        Expects columns: quote_id, candidate, gold_speaker, label, plus all features.
        """
        X = df[self.feature_names]
        scores = self.predict_proba(X)
        
        # Add scores to the dataframe temporarily
        df_eval = df.copy()
        df_eval['score'] = scores
        
        total_quotes = 0
        correct_quotes = 0
        
        # We need to evaluate if the top-ranked candidate is the gold speaker
        for quote_id, group in df_eval.groupby('quote_id'):
            # Only consider quotes where the gold speaker is actually among the candidates
            if group['label'].sum() > 0:
                total_quotes += 1
                
                # Pick the candidate with the highest score
                best_candidate_idx = group['score'].idxmax()
                best_candidate = group.loc[best_candidate_idx, 'candidate']
                gold_speaker = group.loc[best_candidate_idx, 'gold_speaker']
                
                if best_candidate == gold_speaker:
                    correct_quotes += 1
                    
        accuracy = correct_quotes / total_quotes if total_quotes > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_quotes": correct_quotes,
            "total_solvable_quotes": total_quotes
        }
        
    def get_feature_importance(self) -> dict:
        """
        Returns the logistic regression coefficients mapped to feature names.
        """
        if not self.feature_names:
            return {}
            
        coefficients = self.model.coef_[0]
        importance = {name: float(coef) for name, coef in zip(self.feature_names, coefficients)}
        return importance
