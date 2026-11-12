"""Outils d'analyse de la variable cible (Partie 4) : transitions, séquences, couverture des fenêtres, régularité des arrivées."""
from __future__ import annotations
import numpy as np


def transition_matrix(label) -> np.ndarray:
    """Matrice de transition 2x2 entre secondes consécutives : P[i, j] = P(état suivant = j | état courant = i)."""
    l = np.asarray(label, dtype=int); a, b = l[:-1], l[1:]
    counts = np.array([[np.sum((a == i) & (b == j)) for j in (0, 1)] for i in (0, 1)], dtype=float)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)


def run_lengths(label, value: int) -> np.ndarray:
    """Longueurs (secondes) des séquences consécutives égales à `value`."""
    l = (np.asarray(label, dtype=int) == value).astype(int); d = np.diff(np.r_[0, l, 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def duration_coverage(durations, windows=(10, 30, 60, 120, 300)) -> dict:
    """Part des événements au moins aussi longs que chaque fenêtre : une attaque plus courte que la fenêtre y est diluée."""
    d = np.asarray(durations, dtype=float); return {int(w): float((d >= w).mean()) for w in windows}


def interarrival_cv(starts_s) -> float:
    """Coefficient de variation des délais entre débuts d'événements : 1 ≈ arrivées aléatoires (Poisson), < 1 régulier, > 1 par rafales."""
    s = np.sort(np.asarray(starts_s, dtype=float)); ia = np.diff(s)
    return float(ia.std() / ia.mean()) if len(ia) >= 2 and ia.mean() > 0 else float("nan")


def trivial_classifiers(p: float) -> dict:
    """Performances de détecteurs triviaux pour une prévalence p : montre pourquoi l'accuracy est inutilisable."""
    f1 = lambda pr, rc: 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)
    return {"toujours « normal »": {"accuracy": 1 - p, "précision": float("nan"), "rappel": 0.0, "F1": 0.0},
            "toujours « attaque »": {"accuracy": p, "précision": p, "rappel": 1.0, "F1": f1(p, 1.0)},
            "au hasard (50 %)": {"accuracy": .5, "précision": p, "rappel": .5, "F1": f1(p, .5)}}
