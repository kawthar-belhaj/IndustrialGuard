"""Outils d'EDA réutilisables et testés (typologie des variables, AUC univariée, ACF, spectre, stationnarité...)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from scipy import signal, stats

PROC_NAME = {"P1": "P1 Chaudière", "P2": "P2 Turbine", "P3": "P3 Traitement d'eau", "P4": "P4 Simulation HIL", "DCS": "Contrôleurs DCS (x100x)", "Autre": "Autre"}
TYPE_RULES = [(r"^(FCV|PCV|LCV|TCV|SOL)", "Vanne / actionneur"), (r"^(FT|FIT)", "Débit"), (r"^(PIT|PT)", "Pression"), (r"^(TIT|TT)", "Température"),
              (r"^(LIT|LT|LH|LL|LCP)", "Niveau"), (r"^PP", "Pompe"), (r"^(SIT|VT|VXT|VYT|CO_|RTR)", "Vitesse / vibration"), (r"^B\d", "Variable de boucle DCS"),
              (r"^(HT_|ST_)", "Simulateur HIL"), (r"^(OnOff|AutoGO|ManualGO|Emerg|MSD|TripEx|24Vdc|ATSW)", "État / commande turbine")]


def tag_info(tag: str) -> dict:
    """Classement HEURISTIQUE d'un tag (conventions de nommage observées) : processus, type physique, rôle probable."""
    m = re.match(r"P(\d)_", tag)
    proc = f"P{m.group(1)}" if m else ("DCS" if re.match(r"^x\d{4}_", tag) else "Autre")
    body = re.sub(r"^P\d_", "", tag)
    typ = next((lab for pat, lab in TYPE_RULES if re.match(pat, body)), "Autre / non classé")
    if re.search(r"SETPOINT|SP$", body): role = "consigne"
    elif re.search(r"\d+D$", body) or body.endswith("D") and typ in ("Vanne / actionneur", "Pompe"): role = "commande (demande)"
    elif re.search(r"\d+Z$", body) or body.endswith("Z") and typ == "Vanne / actionneur": role = "retour de position"
    elif typ in ("Débit", "Pression", "Température", "Niveau", "Vitesse / vibration"): role = "mesure (capteur)"
    elif typ in ("Simulateur HIL", "État / commande turbine"): role = "état / simulateur"
    else: role = "autre / non classé"
    return {"processus": PROC_NAME[proc], "type_physique": typ, "rôle_probable": role}


def nature_of(series: pd.Series) -> str:
    n = series.nunique(dropna=True)
    return "constante" if n <= 1 else "binaire" if n == 2 else "discrète (≤10)" if n <= 10 else "continue"


def auc_per_column(X: pd.DataFrame, y) -> pd.Series:
    """AUC (attaque vs normal) de chaque colonne prise seule, par rangs (Mann-Whitney). 0,5 = aucune information ; signée."""
    y = np.asarray(y).astype(bool); n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return pd.Series(np.nan, index=X.columns)
    ranks = X.rank(method="average")
    return (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def acf(x, lags) -> np.ndarray:
    x = np.asarray(x, dtype="float64"); x = x - np.nanmean(x); den = np.nansum(x * x)
    return np.array([np.nansum(x[l:] * x[:-l]) / den if den > 0 else np.nan for l in lags])


def acf_half_life(x, lags=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200)) -> float:
    """Premier retard (secondes, interpolé) où l'autocorrélation passe sous 0,5 ; inf si elle reste au-dessus."""
    a = acf(x, lags)
    for i, v in enumerate(a):
        if np.isfinite(v) and v < .5:
            if i == 0: return float(lags[0])
            l0, l1, a0, a1 = lags[i - 1], lags[i], a[i - 1], v
            return float(l0 + (a0 - .5) * (l1 - l0) / max(a0 - a1, 1e-12))
    return float("inf")


def periodogram_peaks(x, fs: float = 1.0, k: int = 3, min_period_s: float = 4.0) -> list:
    """Périodes dominantes (secondes) et part de la puissance : détecte des cycles (rotation, régulation, journée)."""
    x = np.asarray(x, dtype="float64"); x = x[np.isfinite(x)]
    if len(x) < 64 or np.std(x) == 0: return []
    f, p = signal.periodogram(x - x.mean(), fs=fs, detrend="linear"); f, p = f[1:], p[1:]
    keep = (1 / f) >= min_period_s; f, p = f[keep], p[keep]
    if p.sum() <= 0: return []
    top = np.argsort(p)[::-1][:k]
    return [(float(1 / f[i]), float(p[i] / p.sum())) for i in top]


def stationarity(x) -> dict:
    """ADF (H0 : racine unitaire) et KPSS (H0 : stationnaire). Verdict croisé ; sensibles aux grands échantillons (à lire avec prudence)."""
    import warnings
    from statsmodels.tsa.stattools import adfuller, kpss
    warnings.filterwarnings("ignore")
    x = np.asarray(x, dtype="float64"); x = x[np.isfinite(x)]
    if len(x) < 100 or np.std(x) == 0: return {"verdict": "non évaluable", "adf_p": np.nan, "kpss_p": np.nan}
    try:
        adf_p = float(adfuller(x, autolag="AIC")[1]); kp = float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        return {"verdict": "non évaluable", "adf_p": np.nan, "kpss_p": np.nan}
    a, k = adf_p < .05, kp >= .05
    verdict = "stationnaire" if a and k else "racine unitaire / tendance" if (not a) and (not k) else "stationnaire autour d'une tendance ou d'un régime" if a and not k else "non concluant"
    return {"verdict": verdict, "adf_p": adf_p, "kpss_p": kp}


def bimodality_coefficient(x) -> float:
    """> 0,555 suggère une distribution bimodale (ou à plusieurs modes)."""
    x = np.asarray(x, dtype="float64"); x = x[np.isfinite(x)]; n = len(x)
    if n < 10 or np.std(x) == 0: return float("nan")
    g, k = stats.skew(x), stats.kurtosis(x)
    return float((g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def rest_share(x) -> float:
    """Part des points égaux à la médiane : repère les variables « au repos » avec excursions rares."""
    x = np.asarray(x); return float((x == np.nanmedian(x)).mean())
