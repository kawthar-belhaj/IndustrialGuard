"""F3/F4 - Features temporelles CAUSALES, calculées session par session (jamais à cheval sur deux fichiers ni sur un trou).

Pour chaque variable x et chaque fenêtre w (secondes), à l'instant t :
  - niveau   : x(t)
  - delta_w  : x(t) - x(t-w)          (variation sur w secondes)
  - std_w    : écart-type de x sur [t-w+1, t]
  - dev_w    : x(t) - moyenne de x sur [t-w+1, t]     (écart à la tendance récente)
  - range_w  : max - min de x sur [t-w+1, t]          (amplitude récente)
Seules les données <= t sont utilisées : la feature est calculable en production à l'instant t.
Les (wmax) premières secondes de chaque segment ne produisent pas de ligne (fenêtre incomplète).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    X: pd.DataFrame                     # (n_échantillons, n_features) float32
    meta: pd.DataFrame                  # timestamp, session, pos (seconde dans le segment), label (au pas de 1 s)
    y_streams: dict                     # segment -> vecteur 0/1 à 1 Hz (toute la durée du segment)
    stride: int
    wmax: int
    windows: tuple = field(default_factory=tuple)
    families: tuple = field(default_factory=tuple)

    def order(self) -> list:
        return list(dict.fromkeys(self.meta["session"]))


def feature_columns(feats, windows=(10, 30, 60), families=("level", "delta", "std")) -> list:
    cols = []
    if "level" in families:
        cols += [f"{c}|lvl" for c in feats]
    if "delta" in families:
        cols += [f"{c}|d{w}" for w in windows for c in feats]
    if "std" in families:
        cols += [f"{c}|s{w}" for w in windows for c in feats]
    if "dev" in families:
        cols += [f"{c}|dev{w}" for w in windows for c in feats]
    if "range" in families:
        cols += [f"{c}|rng{w}" for w in windows for c in feats]
    return cols


def build_features(df: pd.DataFrame, feats: list, windows=(10, 30, 60), stride: int = 10, families=("level", "delta", "std")) -> FeatureSet:
    families = tuple(families); windows = tuple(windows)
    need_hist = any(f in families for f in ("delta", "std", "dev", "range"))
    wmax = max(windows) if (windows and need_hist) else 0
    has_label = "label" in df.columns
    Xs, metas, ystreams = [], [], {}
    for sess, g in df.groupby("session", sort=False):
        g = g.sort_values("timestamp")
        seg = (g["timestamp"].diff().dt.total_seconds() != 1).cumsum()          # un trou de temps coupe le segment
        n_seg = seg.nunique()
        for k, (_, h) in enumerate(g.groupby(seg, sort=True), 1):
            name = sess if n_seg == 1 else f"{sess}#{k}"
            n = len(h); arr = h[feats].to_numpy(dtype="float64")
            ystreams[name] = h["label"].to_numpy(dtype=int) if has_label else np.zeros(n, dtype=int)
            if n <= wmax:
                continue
            idx = np.arange(wmax, n, stride); parts = []
            if "level" in families:
                parts.append(arr[idx])
            if "delta" in families:
                parts += [arr[idx] - arr[idx - w] for w in windows]
            frame = pd.DataFrame(arr) if any(f in families for f in ("std", "dev", "range")) else None
            if "std" in families:
                parts += [frame.rolling(w, min_periods=w).std().to_numpy()[idx] for w in windows]
            if "dev" in families:
                parts += [arr[idx] - frame.rolling(w, min_periods=w).mean().to_numpy()[idx] for w in windows]
            if "range" in families:
                parts += [(frame.rolling(w, min_periods=w).max() - frame.rolling(w, min_periods=w).min()).to_numpy()[idx] for w in windows]
            Xs.append(np.hstack(parts).astype("float32"))
            metas.append(pd.DataFrame({"timestamp": h["timestamp"].to_numpy()[idx], "session": name, "pos": idx,
                                       "label": ystreams[name][idx]}))
    if not Xs:
        raise ValueError("aucune fenêtre complète : sessions trop courtes pour la plus grande fenêtre")
    X = pd.DataFrame(np.vstack(Xs), columns=feature_columns(feats, windows, families))
    return FeatureSet(X, pd.concat(metas, ignore_index=True), ystreams, stride, wmax, windows, families)
