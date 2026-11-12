"""Partie 7 : mesures de drift (PSI, KS, Wasserstein normalisée, Jensen-Shannon) par rapport à une distribution de référence.

Principe : une seule découpe en classes par variable (déciles de la RÉFÉRENCE), réutilisée pour toutes les fenêtres, afin que les mesures
soient comparables d'une fenêtre à l'autre. Les seuils conventionnels (PSI 0,1 / 0,25) sont des règles empiriques : on les compare toujours
à un seuil NUL calculé sur des fenêtres de la référence elle-même (fluctuation normale).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

EPS = 1e-4


def _fin(x) -> np.ndarray:
    x = np.asarray(x, dtype="float64"); return x[np.isfinite(x)]


class RefDist:
    """Distribution de référence d'une variable (échantillon trié + classes)."""
    def __init__(self, ref, bins: int = 10, max_sample: int = 10_000, seed: int = 42):
        ref = _fin(ref); rng = np.random.default_rng(seed)
        self.n_full = len(ref); self.std = float(ref.std()); u = np.unique(ref)
        self.sample = np.sort(ref if len(ref) <= max_sample else rng.choice(ref, max_sample, replace=False))
        self.discrete = len(u) <= bins
        if self.discrete:
            self.cats = u
        else:
            e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); self.edges = np.r_[-np.inf, e[1:-1], np.inf]
        self.p_ref = self._probs(ref)

    def _probs(self, x) -> np.ndarray:
        x = _fin(x)
        if self.discrete:
            idx = np.searchsorted(self.cats, x); idx = np.clip(idx, 0, len(self.cats) - 1)
            known = self.cats[idx] == x; counts = np.bincount(idx[known], minlength=len(self.cats)).astype(float)
            counts = np.r_[counts, (~known).sum()]                             # dernière classe : valeurs jamais vues dans la référence
        else:
            counts = np.bincount(np.searchsorted(self.edges, x, side="right") - 1, minlength=len(self.edges) - 1).astype(float)
        return counts / max(counts.sum(), 1)

    def _ref_probs_aligned(self) -> np.ndarray:
        return np.r_[self.p_ref, 0.0] if self.discrete and len(self.p_ref) == len(self.cats) else self.p_ref

    def psi(self, cur) -> float:
        pr = np.clip(self._ref_probs_aligned(), EPS, None); pc = np.clip(self._probs(cur), EPS, None)
        pr, pc = pr / pr.sum(), pc / pc.sum(); return float(np.sum((pc - pr) * np.log(pc / pr)))

    def js(self, cur) -> float:
        pr, pc = self._ref_probs_aligned(), self._probs(cur); return float(jensenshannon(pr, pc, base=2))

    def wd(self, cur) -> float:
        cur = _fin(cur); return float(stats.wasserstein_distance(self.sample, cur) / self.std) if self.std > 0 and len(cur) else float("nan")

    def ks(self, cur) -> tuple:
        cur = _fin(cur); r = stats.ks_2samp(self.sample, cur); return float(r.statistic), float(r.pvalue)

    def metrics(self, cur) -> dict:
        d, p = self.ks(cur); return {"psi": self.psi(cur), "ks": d, "ks_p": p, "wd": self.wd(cur), "js": self.js(cur)}


def window_slices(n: int, width_s: int = 3600, freq_s: int = 1, min_fill: float = .9) -> list:
    """Fenêtres fixes non chevauchantes (indices [a, b[) à l'intérieur d'une session ; la dernière est gardée si elle est remplie à min_fill."""
    w = int(width_s / freq_s); return [(a, min(a + w, n)) for a in range(0, n, w) if min(a + w, n) - a >= min_fill * w]


def compute_drift(refs: dict, windows: list, cols: list) -> pd.DataFrame:
    """windows : liste de dicts {window, session, start, data (n × p, colonnes = cols), attack_frac}. Retourne un tableau long (fenêtre × variable)."""
    rows = []
    for w in windows:
        X = w["data"]
        for j, c in enumerate(cols):
            m = refs[c].metrics(X[:, j]); rows.append({"window": w["window"], "session": w["session"], "start": w["start"], "attack_frac": w["attack_frac"], "feature": c, **m})
    return pd.DataFrame(rows)


def null_thresholds(drift_ref: pd.DataFrame, q: float = .95, metrics=("psi", "ks", "wd", "js")) -> pd.DataFrame:
    """Seuil nul par variable et par mesure : quantile q des valeurs observées sur des fenêtres de la RÉFÉRENCE (fluctuation normale)."""
    return drift_ref.groupby("feature")[list(metrics)].quantile(q)


def local_wd(pre: np.ndarray, att: np.ndarray, std_ref: np.ndarray) -> np.ndarray:
    """Distance de Wasserstein (en écarts-types de référence) entre deux échantillons, variable par variable."""
    return np.array([stats.wasserstein_distance(pre[:, j], att[:, j]) / std_ref[j] if std_ref[j] > 0 else np.nan for j in range(pre.shape[1])])
