"""Génère de FAUSSES données au format HAI 23.05 pour tester la chaîne sans télécharger 1 Go.
Injecte volontairement : doublons, trous, NaN, colonne constante, attaques."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def make(n, seed, attacks=False, start="2023-01-01"):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="1s")
    s = np.arange(n)
    df = pd.DataFrame({"timestamp": t,
        "P1_B2004": 10 + np.sin(s / 50) + rng.normal(0, .1, n),
        "P1_B3004": 30 + 2 * np.cos(s / 80) + rng.normal(0, .2, n),
        "P2_SIT01": 500 + 5 * np.sin(s / 30) + rng.normal(0, 1, n),
        "P3_LH": 50 + rng.normal(0, .5, n),
        "P4_ST_PS": np.full(n, 3.0)})                       # constante
    label = np.zeros(n, dtype=int)
    if attacks:
        for a in range(n // 5, n, n // 4):
            df.loc[a:a + 120, "P1_B2004"] += 4
            df.loc[a:a + 120, "P2_SIT01"] -= 25
            label[a:a + 121] = 1
    return df, label

def main(out="data/raw"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for i in range(1, 5):
        df, _ = make(4000, i, start=f"2023-01-0{i}")
        df.loc[100:105, "P1_B3004"] = np.nan
        df = pd.concat([df, df.iloc[50:55]])                  # doublons
        df = df.drop(index=range(900, 920))                   # trou de 20 s
        df.to_csv(out / f"hai-train{i}.csv", index=False)
    for i in (1, 2):
        df, lab = make(4000, 10 + i, attacks=True, start=f"2023-02-0{i}")
        df.to_csv(out / f"hai-test{i}.csv", index=False)
        pd.DataFrame({"timestamp": df["timestamp"], "label": lab}).to_csv(out / f"label-test{i}.csv", index=False)

if __name__ == "__main__":
    main(*sys.argv[1:])
