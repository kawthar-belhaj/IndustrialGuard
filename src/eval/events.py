"""Évaluation PAR ÉVÉNEMENT (F11) : eTaPR officiel + rappel par attaque, délai de détection, fausses alertes par heure.

ATTENTION convention eTaPR : 0 = normal, 1 = anomalie (le README du dépôt dit « 1 / -1 » mais le code lit 0 / 1 :
tester à l'envers donne des scores parfaits trompeurs). La bibliothèque PLANTE si aucune anomalie n'est prédite : géré ici.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def smooth_scores(scores, sessions, k: int = 3) -> np.ndarray:
    """Médiane glissante CAUSALE (k derniers échantillons) par session : écarte les pics isolés sans regarder le futur."""
    s = pd.Series(np.asarray(scores, dtype="float64"))
    return s.groupby(np.asarray(sessions)).transform(lambda x: x.rolling(k, min_periods=1).median()).to_numpy()


def flags_to_streams(fs, flags) -> dict:
    """Décision prise à l'instant t (échantillon) -> alarme sur [t, t+stride) à 1 Hz, par segment."""
    flags = np.asarray(flags, dtype=bool); out = {k: np.zeros(len(v), dtype=int) for k, v in fs.y_streams.items()}
    sess, pos = fs.meta["session"].to_numpy(), fs.meta["pos"].to_numpy()
    for i in np.flatnonzero(flags):
        out[sess[i]][pos[i]:pos[i] + fs.stride] = 1
    return out


def concat_streams(fs, p_streams):
    """Concatène (ordre d'apparition) avec un séparateur normal de 1 s : deux sessions ne fusionnent jamais."""
    y, p = [], []
    for s in fs.order() + [k for k in fs.y_streams if k not in fs.order()]:
        y += [fs.y_streams[s], [0]]; p += [p_streams[s], [0]]
    return np.concatenate(y).astype(int), np.concatenate(p).astype(int)


def runs(stream) -> list:
    st = np.asarray(stream, dtype=int); d = np.diff(np.r_[0, st, 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def etapr_scores(y, p, theta_p: float = 0.5, theta_r: float = 0.1, delta: float = 0.0) -> dict:
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int)
    if y.sum() == 0:
        raise ValueError("aucune attaque dans le flux de référence")
    if p.sum() == 0:                                            # la bibliothèque plante dans ce cas
        return {"eTaR": 0.0, "eTaP": 0.0, "etapr_f1": 0.0, "n_false_alarm_etapr": 0}
    try:
        from eTaPR_pkg import etapr
    except ImportError as e:
        raise ImportError("eTaPR n'est pas installé : pip install \"eTaPR @ https://raw.githubusercontent.com/saurf4ng/eTaPR/main/eTaPR-22.6.1-py3-none-any.whl\"") from e
    r = etapr.evaluate_w_streams(y.tolist(), p.tolist(), theta_p, theta_r, delta)
    return {"eTaR": float(r["eTaR"]), "eTaP": float(r["eTaP"]), "etapr_f1": float(r["f1"]), "n_false_alarm_etapr": int(r["N False Alarm"])}


def event_report(y, p, grace_s: int = 60) -> dict:
    """Une attaque est détectée si une alarme tombe dans [début, fin + grace_s]. Une alarme sans aucune attaque (marge incluse) est une fausse alerte."""
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=int); n = len(y)
    atk, alm = runs(y), runs(p); delays, detected, detail = [], 0, []
    for s, e in atk:
        hit = np.flatnonzero(p[s:min(n, e + grace_s + 1)])
        if len(hit): detected += 1; delays.append(int(hit[0]))
        detail.append({"start": int(s), "end": int(e), "duration_s": int(e - s + 1), "detected": bool(len(hit)), "delay_s": int(hit[0]) if len(hit) else None})
    fa = sum(1 for s, e in alm if not any(s <= ae + grace_s and e >= as_ for as_, ae in atk))
    normal_h = max(float((y == 0).sum()) / 3600.0, 1e-9)
    return {"n_attacks": len(atk), "n_detected": detected, "event_recall": detected / max(len(atk), 1),
            "median_delay_s": float(np.median(delays)) if delays else float("nan"), "n_alarm_segments": len(alm),
            "n_false_alarm_segments": fa, "fa_per_hour": fa / normal_h, "per_attack_delay_s": delays, "attacks": detail}


def point_metrics(labels, scores) -> dict:
    labels = np.asarray(labels, dtype=int)
    if labels.min() == labels.max():
        return {"roc_auc": float("nan"), "avg_precision": float("nan")}
    return {"roc_auc": float(roc_auc_score(labels, scores)), "avg_precision": float(average_precision_score(labels, scores))}


def evaluate(fs, scores, thr: float, theta_p=0.5, theta_r=0.1, grace_s=60) -> dict:
    flags = np.asarray(scores) > thr
    y, p = concat_streams(fs, flags_to_streams(fs, flags))
    return {"threshold": float(thr), **etapr_scores(y, p, theta_p, theta_r), **event_report(y, p, grace_s)}


# ------------------------------------------------------------------ stratégies de seuil (F6, hypothèse H6)
def _grid(scores, lo=.90, hi=.99995, n=60):
    return np.unique(np.quantile(scores, np.linspace(lo, hi, n)))


def thr_quantile(scores_normal, q: float = .999) -> float:
    """Seuil = quantile q des scores d'un jeu NORMAL (aucune attaque nécessaire)."""
    return float(np.quantile(scores_normal, q))


def thr_fa_budget(fs_normal, scores_normal, budget_per_hour: float = 1.0) -> float:
    """Plus petit seuil tel que le nombre de fausses alertes (segments d'alarme) par heure NORMALE reste <= budget."""
    for thr in _grid(scores_normal, .5, .99999, 200):
        flags = np.asarray(scores_normal) > thr
        y, p = concat_streams(fs_normal, flags_to_streams(fs_normal, flags))
        if len(runs(p)) / max(len(y) / 3600.0, 1e-9) <= budget_per_hour:
            return float(thr)
    return float(np.max(scores_normal))


def thr_best_f1(fs_val, scores_val, theta_p=0.5, theta_r=0.1, grace_s=60) -> float:
    """Seuil maximisant l'eTaPR-F1 sur la VALIDATION (dépend de peu d'attaques : sur-ajustement possible, hypothèse H6)."""
    best, best_thr = -1.0, float(np.max(scores_val))
    for thr in _grid(scores_val):
        f1 = evaluate(fs_val, scores_val, thr, theta_p, theta_r, grace_s)["etapr_f1"]
        if f1 >= best: best, best_thr = f1, float(thr)
    return best_thr
