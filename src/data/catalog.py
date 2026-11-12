"""Segments d'attaque et recoupement avec le catalogue officiel (summary_label*.txt)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd


def segments(df: pd.DataFrame) -> pd.DataFrame:
    """Segments contigus de label=1. Un segment couvre [début, fin] INCLUS (durée = fin - début + 1 s)."""
    l = df["label"].to_numpy()
    st = np.flatnonzero(np.diff(np.r_[0, l]) == 1)
    en = np.flatnonzero(np.diff(np.r_[l, 0]) == -1)
    s = pd.DataFrame({"i0": st, "i1": en, "début": df["timestamp"].iloc[st].to_numpy(),
                      "fin": df["timestamp"].iloc[en].to_numpy(), "durée_s": en - st + 1})
    s["pause_avant_s"] = np.r_[np.nan, (st[1:] - en[:-1] - 1)] if len(s) else []
    return s


_ROW = re.compile(r"\[\s*(\d+)\]\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\d+):(\d{2}):(\d{2})")


def parse_summary(path) -> tuple[pd.DataFrame, dict]:
    """Lit un summary_label*.txt -> (tableau des attaques listées, en-tête)."""
    txt = Path(path).read_text(errors="replace")
    rows = [(int(i), pd.Timestamp(a), pd.Timestamp(b), int(h) * 3600 + int(m) * 60 + int(s)) for i, a, b, h, m, s in _ROW.findall(txt)]
    df = pd.DataFrame(rows, columns=["index", "start", "end", "duration_s"])
    head = {"attacks_listed": len(df), "total_listed_s": int(df["duration_s"].sum())}
    m = re.search(r"Attacks\s*:\s*(\d+)\s*times", txt)
    head["attacks_header"] = int(m.group(1)) if m else None
    m = re.search(r"Total\s*:\s*(\d+):(\d{2}):(\d{2})", txt)
    head["total_header_s"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None
    return df, head


def crosscheck(seg: pd.DataFrame, summ: pd.DataFrame, tol_s: float = 2.0) -> tuple[pd.DataFrame, list[int]]:
    """Pour chaque attaque officielle, trouve le segment de labels le plus proche et mesure l'écart début/fin (s)."""
    res = []
    for r in summ.itertuples():
        j = int((seg["début"] - r.start).abs().dt.total_seconds().idxmin())
        s = seg.loc[j]
        res.append({"id": r.index, "seg": j, "start_diff_s": (s["début"] - r.start).total_seconds(), "end_diff_s": (s["fin"] - r.end).total_seconds()})
    out = pd.DataFrame(res)
    out["ok"] = (out["start_diff_s"].abs() <= tol_s) & (out["end_diff_s"].abs() <= tol_s)
    extra = sorted(set(seg.index) - set(out["seg"]))
    return out, extra
