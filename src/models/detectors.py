"""Détecteurs de référence (progression « du plus simple au plus complexe » : chaque modèle doit justifier sa complexité).

 - maxz     : score = max |z| sur les variables (seuil univarié, aucun apprentissage) ;
 - pca      : erreur de reconstruction d'une ACP linéaire (autoencodeur linéaire) ;
 - iforest  : Isolation Forest (baseline du cahier des charges).
Tous : fit(X_train) sur données NORMALES uniquement ; score(X) -> plus grand = plus anormal.
"""
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class MaxZ:
    name = "maxz"
    def fit(self, X): return self
    def score(self, X): return np.max(np.abs(np.asarray(X, dtype="float64")), axis=1)       # X déjà en z-scores (variables « niveau »)
    def params(self): return {}


class ScaledPCA:
    name = "pca"
    def __init__(self, n_components=0.95, seed=42): self.n_components, self.seed = n_components, seed
    def fit(self, X):
        self.scaler = StandardScaler().fit(X); self.pca = PCA(n_components=self.n_components, svd_solver="full", random_state=self.seed).fit(self.scaler.transform(X)); return self
    def score(self, X):
        Z = self.scaler.transform(X); R = self.pca.inverse_transform(self.pca.transform(Z)); return ((Z - R) ** 2).sum(axis=1)
    def params(self): return {"n_components": self.n_components, "n_components_fitted": int(self.pca.n_components_)}


class IForest:
    name = "iforest"
    def __init__(self, n_estimators=200, max_samples=256, seed=42): self.kw = dict(n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=-1)
    def fit(self, X): self.model = IsolationForest(**self.kw).fit(X); return self
    def score(self, X): return -self.model.score_samples(X)
    def params(self): return {k: v for k, v in self.kw.items() if k != "n_jobs"}


def make_detector(name: str, seed: int = 42):
    return {"maxz": lambda: MaxZ(), "pca": lambda: ScaledPCA(seed=seed), "iforest": lambda: IForest(seed=seed)}[name]()
