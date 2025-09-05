# custom_transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class RiskClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, random_state=42, column_name="risk_level"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.column_name = column_name
        self.kmeans_ = None
        self.cluster_map_ = None

    def fit(self, X, y=None):
        X_np = X.values if hasattr(X, "values") else np.asarray(X)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        clusters = self.kmeans_.fit_predict(X_np)

        if y is None:
            order = np.argsort(np.bincount(clusters))
            self.cluster_map_ = {c: i for i, c in enumerate(order)}
        else:
            df = pd.DataFrame({"cluster": clusters, "default": np.asarray(y)})
            cluster_risk = df.groupby("cluster")["default"].mean().sort_values()
            self.cluster_map_ = {c: i for i, c in enumerate(cluster_risk.index)}

        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        X_np = X.values if is_df else np.asarray(X)

        clusters = self.kmeans_.predict(X_np)
        risk_levels = np.array([self.cluster_map_[c] for c in clusters]).reshape(-1, 1)

        if is_df:
            X_out = X.copy()
            X_out[self.column_name] = risk_levels.ravel()
            return X_out
        else:
            return np.hstack([X_np, risk_levels])
