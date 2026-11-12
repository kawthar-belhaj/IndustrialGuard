"""Nettoyage, protocoles de découpage, normalisation. Utilisé à l'identique par le notebook, le pipeline DVC et (Phase 3) l'API.

Deux protocoles (voir notebook 1, §4.3) :
  * "sessions" : train = train1-3 | val_normal = train4 | val = test1 | test = test2.
                 Sessions distinctes SANS chevauchement, mais PAS strictement chronologiques
                 (train3/train4 sont postérieurs à test2). Mesure la généralisation à de nouvelles sessions.
  * "chrono"   : tout ce qui sert à décider (train, val_normal, val) est ANTÉRIEUR à test2. Simule une mise en production ;
                 train3 + train4 (postérieurs) forment un flux de production normal pour la démo de drift.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data.io import sensor_columns


def clean_file(df: pd.DataFrame, freq_s: int = 1, max_ffill_s: int = 5) -> pd.DataFrame:
    """Trie, dé-duplique, régularise à 1 pas/s, comble les PETITS trous. Un label ambigu n'est jamais deviné.

    - doublons de timestamp avec labels contradictoires -> erreur ;
    - ligne insérée : capteurs ffill (<= max_ffill_s) ; label conservé seulement si les deux voisins s'accordent, sinon ligne supprimée ;
    - les grands trous restent des trous (lignes supprimées, jamais inventées).
    """
    has_label = "label" in df.columns
    if has_label:
        conflict = df.groupby("timestamp")["label"].nunique()
        if (conflict > 1).any():
            raise ValueError(f"{int((conflict > 1).sum())} timestamps dupliqués avec des labels contradictoires")
    orig_idx = pd.DatetimeIndex(df["timestamp"].unique())
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first").set_index("timestamp")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq=f"{freq_s}s"))
    cols = [c for c in df.columns if c != "label"]
    lim = max(1, int(max_ffill_s / freq_s))
    df[cols] = df[cols].ffill(limit=lim)
    keep = df[cols].notna().all(axis=1)
    if has_label:
        f, b = df["label"].ffill(limit=lim), df["label"].bfill(limit=lim)
        df["label"] = f.where(f == b)
        keep &= df["label"].notna()
    inserted_kept = int((keep & ~df.index.isin(orig_idx)).sum())
    df = df[keep]
    if has_label:
        df["label"] = df["label"].astype(int)
    df.index.name = "timestamp"
    out = df.reset_index()
    out.attrs["inserted_kept"] = inserted_kept
    return out


def _tag(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy(); out["session"] = name; return out


def apply_embargo(clean: dict, embargo_s: int = 600, contiguity_s: float = 5.0) -> tuple[dict, list]:
    """Purge aux frontières CONTIGUËES : un fichier d'apprentissage qui démarre (ou finit) à moins de `contiguity_s` secondes
    d'un fichier de validation/test prolonge l'état du procédé (autocorrélation ~0,5 à 600 s). On retire `embargo_s` secondes
    côté apprentissage pour éviter une fuite d'information à la frontière. Retourne (clean purgé, journal)."""
    out, log = dict(clean), []
    if embargo_s <= 0:
        return out, log
    for tr in ("train1", "train2", "train3", "train4"):
        t = out[tr]
        for te in ("test1", "test2"):
            e = clean[te]
            if 0 <= (t["timestamp"].min() - e["timestamp"].max()).total_seconds() <= contiguity_s:      # t suit e
                t = t.iloc[embargo_s:].reset_index(drop=True); log.append({"fichier": tr, "côté": "début", "lignes_retirées": embargo_s, "raison": f"contigu à la fin de {te}"})
            if 0 <= (e["timestamp"].min() - t["timestamp"].max()).total_seconds() <= contiguity_s:      # t précède e
                t = t.iloc[:-embargo_s].reset_index(drop=True); log.append({"fichier": tr, "côté": "fin", "lignes_retirées": embargo_s, "raison": f"contigu au début de {te}"})
        out[tr] = t
    return out, log


def split_frames(clean: dict, protocol: str, holdout_hours: float = 10.0, embargo_s: int = 0) -> dict:
    """clean : dict nom -> DataFrame nettoyé (train1..4, test1, test2). Retourne les jeux bruts (non normalisés)."""
    cat = lambda *x: pd.concat(x, ignore_index=True)
    c = {k: _tag(v, k) for k, v in clean.items()}          # colonne « session » : le fenêtrage ne devra JAMAIS enjamber deux sessions
    if protocol == "sessions":
        return {"train": cat(c["train1"], c["train2"], c["train3"]), "val_normal": c["train4"], "val": c["test1"], "test": c["test2"]}
    if protocol == "chrono":
        h = int(holdout_hours * 3600)
        t2 = c["train2"]
        if len(t2) <= h + embargo_s:
            raise ValueError("train2 trop court pour le holdout chronologique")
        fr = {"train": cat(c["train1"], t2.iloc[:-h]), "val_normal": t2.iloc[len(t2) - h + embargo_s:].reset_index(drop=True),
              "val": c["test1"], "test": c["test2"], "stream": cat(c["train3"], c["train4"])}
        decide_max = max(fr[k]["timestamp"].max() for k in ("train", "val_normal", "val"))
        if not decide_max < fr["test"]["timestamp"].min():
            raise ValueError("protocole chrono : des données de décision ne précèdent pas le test")
        return fr
    raise ValueError(f"protocole inconnu : {protocol}")


@dataclass
class Prepared:
    protocol: str
    splits: dict
    scaler: StandardScaler
    feats: list
    dropped: list
    unstable: dict
    band_rules: dict
    clip_stats: dict
    ref_stats: pd.DataFrame
    spans: dict = field(default_factory=dict)


def transform(df: pd.DataFrame, scaler: StandardScaler, feats: list, clip_sigma: float | None = None) -> pd.DataFrame:
    z = scaler.transform(df[feats])
    if clip_sigma:
        z = np.clip(z, -clip_sigma, clip_sigma)
    out = pd.DataFrame(z.astype("float32"), columns=feats)
    out.insert(0, "timestamp", df["timestamp"].to_numpy())
    if "label" in df.columns:
        out["label"] = df["label"].to_numpy()
    if "session" in df.columns:
        out["session"] = df["session"].to_numpy()
    return out


def prepare_protocol(name: str, frames: dict, clip_sigma: float | None = 10, exclude=(), unstable_z: float = 50,
                     unstable_policy: str = "keep_and_flag") -> Prepared:
    """Tout ce qui est APPRIS (colonnes, scaler, clipping) vient du train seul."""
    train = frames["train"]
    common = set(sensor_columns(train))
    for k in ("val_normal", "val", "test"):
        common &= set(sensor_columns(frames[k]))
    feats = [c for c in sensor_columns(train) if c in common and c not in set(exclude) and train[c].std() > 0]
    scaler = StandardScaler().fit(train[feats])

    zmax = np.abs(scaler.transform(frames["val_normal"][feats])).max(axis=0)
    unstable = {f: float(v) for f, v in zip(feats, zmax) if v > unstable_z}
    band_rules = {}
    if unstable and unstable_policy == "rule":
        normal = pd.concat([train, frames["val_normal"]])
        for f in unstable:
            lo, hi = float(normal[f].min()), float(normal[f].max()); m = .05 * (hi - lo)
            band_rules[f] = {"min": lo - m, "max": hi + m}
        feats = [f for f in feats if f not in unstable]
        scaler = StandardScaler().fit(train[feats])

    zt = np.abs(scaler.transform(train[feats]))
    clipped = (zt > clip_sigma) if clip_sigma else np.zeros_like(zt, dtype=bool)
    by_var = pd.Series(clipped.sum(axis=0), index=feats).sort_values(ascending=False)
    clip_stats = {"total": int(clipped.sum()), "share_pct": float(100 * clipped.mean()), "top_vars": by_var[by_var > 0].head(10).to_dict()}

    splits = {k: transform(frames[k], scaler, feats, clip_sigma if k == "train" else None) for k in frames}
    spans = {k: (frames[k]["timestamp"].min(), frames[k]["timestamp"].max(), len(frames[k])) for k in frames}
    ref = train[feats].describe().T[["mean", "std", "min", "max"]]
    return Prepared(name, splits, scaler, feats, sorted(set(sensor_columns(train)) - set(feats)), unstable, band_rules, clip_stats, ref, spans)


def save_prepared(p: Prepared, out_dir) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for k, df in p.splits.items():
        df.to_parquet(out / f"{k if k != 'stream' else 'stream_normal'}.parquet", index=False)
    joblib.dump(p.scaler, out / "scaler.joblib")
    json.dump(p.feats, open(out / "features.json", "w"), indent=2)
    p.ref_stats.to_json(out / "train_reference_stats.json", orient="index", indent=2)
    meta = {"protocol": p.protocol, "n_features": len(p.feats), "dropped_columns": p.dropped, "unstable_variables": p.unstable,
            "band_rules": p.band_rules, "clip": p.clip_stats, "rows": {k: len(v) for k, v in p.splits.items()},
            "attack_ratio": {k: float(p.splits[k]["label"].mean()) for k in ("val", "test")}}
    json.dump(meta, open(out / "meta.json", "w"), indent=2, default=str)
    return meta
