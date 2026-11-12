"""Lecture STRICTE des CSV HAI.

Principes :
  * la version du dataset est exigée (aucun repli silencieux vers une autre version) ;
  * toute ambiguïté (0 ou plusieurs fichiers candidats) est une erreur ;
  * un label manquant n'est JAMAIS converti en « normal » : on s'arrête.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TIME_NAMES = ("timestamp", "time", "datetime", "date")
LABEL_NAMES = ("label", "attack")
DATA_KEYS = ["hai-train1", "hai-train2", "hai-train3", "hai-train4", "hai-test1", "hai-test2"]
LABEL_KEYS = ["label-test1", "label-test2"]


def find_hai_files(data_root, version: str = "23.05", allow_unversioned: bool = False) -> dict[str, Path]:
    """Retrouve les 8 fichiers de la version EXACTE. Lève FileNotFoundError si absent ou ambigu."""
    root = Path(data_root)
    csvs = sorted(root.rglob("*.csv"))
    found, problems = {}, []
    for key in DATA_KEYS + LABEL_KEYS:
        hits = []
        for p in csvs:
            if p.name.lower() != f"{key}.csv":
                continue
            parts = [x.lower() for x in p.relative_to(root).parts[:-1]]
            if any("haiend" in x for x in parts):
                continue                                   # HAIEnd (225 tags) n'est pas utilisé
            if allow_unversioned or version in "/".join(parts):
                hits.append(p)
        if not hits:
            problems.append(f"{key}.csv introuvable pour la version « {version} » sous {root}")
        elif len(hits) > 1:
            problems.append(f"{key}.csv : {len(hits)} correspondances ambiguës : {[str(h) for h in hits]}")
        else:
            found[key] = hits[0]
    if problems:
        raise FileNotFoundError("Sélection des fichiers impossible :\n  - " + "\n  - ".join(problems))
    return found


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _find(df: pd.DataFrame, names) -> str | None:
    return next((c for c in df.columns if c.lower() in names), None)


def align_labels(df: pd.DataFrame, lab: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Associe les labels aux mesures. Leçon HAI 23.05 : label-test2.csv a des timestamps tronqués à la MINUTE.

    1. timestamps de labels uniques -> jointure ; toute ligne sans label est une ERREUR (jamais 0 par défaut) ;
    2. sinon, même nombre de lignes -> alignement POSITIONNEL, avec vérifications : données et labels triés,
       même nombre de lignes par minute, concordance à la minute à 100 % ;
    3. sinon -> erreur.
    L'ordre à l'intérieur d'une minute n'est pas vérifiable ici : il est recoupé avec le catalogue officiel
    (src.data.catalog.crosscheck, au niveau de la seconde).
    """
    ltime = _find(lab, TIME_NAMES)
    lcol = _find(lab, LABEL_NAMES) or [c for c in lab.columns if c != ltime][-1]
    y = pd.to_numeric(lab[lcol], errors="coerce")
    if y.isna().any():
        raise ValueError(f"{int(y.isna().sum())} labels manquants ou non numériques : refus de les traiter comme « normal »")
    y = y.astype(int).to_numpy()
    if not set(np.unique(y)) <= {0, 1}:
        raise ValueError(f"labels hors {{0,1}} : {sorted(set(np.unique(y)))}")
    lts = pd.to_datetime(lab[ltime]) if ltime else None
    info = {"data_rows": len(df), "label_rows": len(lab), "label_unique_timestamps": int(lts.nunique()) if lts is not None else None}

    if lts is not None and lts.is_unique:
        m = df[["timestamp"]].merge(pd.DataFrame({"timestamp": lts, "label": y}), on="timestamp", how="left")
        if len(m) != len(df):
            raise ValueError("la jointure des labels a modifié le nombre de lignes")
        unmatched = int(m["label"].isna().sum())
        info.update(mode="jointure timestamp", unmatched_rows=unmatched)
        if unmatched:
            raise ValueError(f"{unmatched} lignes de mesures sans label : refus de les considérer comme normales")
        return m["label"].astype(int).to_numpy(), info

    if len(lab) == len(df):
        info["mode"] = "positionnel"
        if lts is not None:
            if not df["timestamp"].is_monotonic_increasing:
                raise ValueError("données non triées : alignement positionnel non fiable")
            if not lts.is_monotonic_increasing:
                raise ValueError("timestamps de labels non triés : alignement positionnel non fiable")
            dm, lm = df["timestamp"].dt.floor("min"), lts.dt.floor("min")
            cd, cl = dm.value_counts().sort_index(), lm.value_counts().sort_index()
            if not (np.array_equal(cd.index.values, cl.index.values) and np.array_equal(cd.values, cl.values)):
                raise ValueError("le nombre de lignes par minute diffère entre données et labels")
            conc = float((dm.to_numpy() == lm.to_numpy()).mean())
            info.update(minutes=int(len(cd)), rows_per_minute_equal=True, labels_sorted=True, minute_concordance_pct=100 * conc)
            if conc < 1.0:
                raise ValueError(f"concordance à la minute incomplète ({100*conc:.3f} %)")
        return y, info
    raise ValueError(f"labels ({len(lab)} lignes, timestamps non uniques) incompatibles avec les données ({len(df)} lignes)")


def load_hai_csv(path, labels_path=None, info_sink: dict | None = None, float32: bool = True) -> pd.DataFrame:
    df = _clean_cols(pd.read_csv(path))
    tcol = _find(df, TIME_NAMES)
    if tcol is None:
        raise ValueError(f"{path}: aucune colonne temporelle trouvée parmi {TIME_NAMES}")
    df = df.rename(columns={tcol: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    inline = _find(df, LABEL_NAMES)
    if inline and inline != "label":
        df = df.rename(columns={inline: "label"})
    if labels_path is not None:
        y, info = align_labels(df, _clean_cols(pd.read_csv(labels_path)))
        df["label"] = y
        if info_sink is not None:
            info_sink[Path(path).stem] = info
    if "label" in df.columns:
        if df["label"].isna().any():
            raise ValueError(f"{path}: labels manquants")
        df["label"] = df["label"].astype(int)
    if float32:
        cols = sensor_columns(df)
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    return df


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("timestamp", "label", "session")]
