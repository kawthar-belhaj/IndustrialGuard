"""F1 - Contrôle qualité des données : manquants, doublons, aberrants, continuité temporelle."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.data.io import find_hai_files, load_hai_csv, sensor_columns


def validate_frame(df: pd.DataFrame, name: str, freq_s: int = 1, robust_z: float = 8.0, max_missing_ratio: float = 0.05) -> dict:
    cols = sensor_columns(df); ts = df["timestamp"]
    dt = ts.diff().dt.total_seconds().dropna()
    missing = df[cols].isna().mean()
    x = df[cols]; med = x.median(); mad = (x - med).abs().median().replace(0, np.nan)
    out_ratio = (((x - med).abs() / (1.4826 * mad)) > robust_z).mean().dropna().sort_values(ascending=False)
    gaps = dt[dt > freq_s]
    return {"file": name, "rows": int(len(df)), "sensor_columns": len(cols), "time_start": str(ts.min()), "time_end": str(ts.max()),
            "is_sorted": bool(ts.is_monotonic_increasing), "duplicate_timestamps": int(ts.duplicated().sum()),
            "duplicate_rows": int(df.drop(columns=["timestamp"]).duplicated().sum()), "missing_cells": int(df[cols].isna().sum().sum()),
            "columns_over_missing_limit": missing[missing > max_missing_ratio].round(4).to_dict(),
            "constant_columns": [c for c in cols if df[c].nunique(dropna=True) <= 1],
            "time_gaps": {"count": int(len(gaps)), "max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
                          "missing_seconds": float((gaps - freq_s).sum()) if len(gaps) else 0.0},
            "outlier_ratio_top5 (MAD, indicatif)": out_ratio.head(5).round(5).to_dict(),
            "label_ratio": float(df["label"].mean()) if "label" in df.columns else None}


def run(params_path: str = "params.yaml") -> dict:
    p = yaml.safe_load(open(params_path)); d, v = p["data"], p["validate"]
    files = find_hai_files(d["raw_dir"], d.get("version_tag", "23.05"), d.get("allow_unversioned", False))
    report = {"dataset_version": d.get("version_tag"), "label_alignment": {}, "files": []}
    for k in ("hai-train1", "hai-train2", "hai-train3", "hai-train4", "hai-test1", "hai-test2"):
        lab = files.get("label-" + k.split("-")[1]) if "test" in k else None
        df = load_hai_csv(files[k], lab, report["label_alignment"])
        report["files"].append(validate_frame(df, k, d["freq_seconds"], v["outlier_robust_z"], v["max_missing_ratio"]))
    Path(d["report_path"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(d["report_path"], "w"), indent=2, default=str)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", default="params.yaml")
    for f in run(ap.parse_args().params)["files"]:
        print(f"{f['file']:<12} rows={f['rows']:>8} dup_ts={f['duplicate_timestamps']:>4} missing={f['missing_cells']:>6} gaps={f['time_gaps']['count']:>3}")
