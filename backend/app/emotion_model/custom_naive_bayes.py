import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomMultinomialNB(BaseEstimator, ClassifierMixin):
    """
    Custom implementation of Multinomial Naive Bayes.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        X: Sparse matrix of shape (n_samples, n_features)
        y: Array-like of shape (n_samples,)
        """
        # Get unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        # Use numpy array for boolean indexing to satisfy scipy.sparse
        y_np = np.array(y)
        
        # Calculate stats for each class
        for idx, c in enumerate(self.classes_):
            # Mask for current class
            X_c = X[y_np == c]
            
            # Count class occurrences (Prior)
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])
            
            # Count feature occurrences (Likelihood)
            # Sum columns to get total count of each feature for this class
            feature_counts = np.array(X_c.sum(axis=0)).flatten()
            
            # Add smoothing (alpha)
            smoothed_counts = feature_counts + self.alpha
            
            # Normalize by total count of all features in this class (+ alpha * n_features)
            total_count = smoothed_counts.sum()
            
            self.feature_log_prob_[idx] = np.log(smoothed_counts / total_count)
            
        return self

    def predict(self, X):
        """
        X: Sparse matrix of shape (n_samples, n_features)
        Returns: predicted labels
        """
        # Log probability = X * feature_log_prob_T + class_log_prior_
        # X is (n_samples, n_features)
        # feature_log_prob_.T is (n_features, n_classes)
        # Result is (n_samples, n_classes)
        
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        
        # Argmax to get class index
        return self.classes_[np.argmax(jll, axis=1)]
