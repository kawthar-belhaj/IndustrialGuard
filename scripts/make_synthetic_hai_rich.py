"""Jeu de DÉMO riche au format HAI 23.05 (30 tags, 4 processus, corrélations, attaques variées).
Sert uniquement à tester le notebook sans télécharger HAI. NE PAS utiliser pour des résultats."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

TAGS = {  # tag -> (poids charge, poids bruit, offset, échelle)
 "P1_FT01":(1.0,.3,20,2),"P1_FT02":(.9,.3,18,2),"P1_PIT01":(.6,.2,5,.5),"P1_PIT02":(.5,.25,4.5,.5),
 "P1_TIT01":(.4,.2,60,3),"P1_TIT02":(.35,.25,58,3),"P1_FCV01D":(1.2,.2,50,5),"P1_FCV01Z":(1.2,.05,50,5),
 "P1_B2004":(.3,.4,10,1),"P1_B3004":(.2,.4,30,2),"P1_LIT01":(-.5,.3,40,4),
 "P2_SIT01":(1.5,.2,500,20),"P2_SIT02":(1.4,.25,498,20),"P2_VT01":(.2,.8,2,.3),"P2_CO_rpm":(1.0,.3,3000,50),
 "P3_LIT01":(.7,.3,55,5),"P3_FIT01":(.8,.3,12,1),"P3_PIT01":(.4,.3,3,.3),
 "P4_HT_PO":(.9,.2,100,8),"P4_ST_PO":(1.0,.2,95,8),"P4_ST_LD":(.9,.3,70,6),
}
BINARY = ["P1_PP01AD", "P2_OnOff", "P2_AutoGO"]

def ar1(rng, n, phi=.98):
    e = rng.normal(0, 1, n); x = np.zeros(n)
    for i in range(1, n): x[i] = phi * x[i-1] + e[i]
    return x / x.std()

def make(n, seed, start, shift=0.0, attacks=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq="1s"); s = np.arange(n)
    load = np.sin(2*np.pi*s/3600) + .5*np.sin(2*np.pi*s/900) + .3*ar1(rng, n, .995)
    d = {"timestamp": t}
    for k, (wl, wn, off, sc) in TAGS.items():
        d[k] = off + sc*(wl*load + wn*ar1(rng, n, .9)) + shift*sc
    d["P1_FCV01Z"] = d["P1_FCV01D"] + rng.normal(0, .1, n)
    d["P1_PP01AD"] = (load > -.3).astype(int); d["P2_OnOff"] = 1; d["P2_AutoGO"] = (load > 0).astype(int)
    d["P4_ST_PS"] = 3.0
    df = pd.DataFrame(d); label = np.zeros(n, dtype=int)
    if attacks:
        tags = [c for c in df.columns if c not in ("timestamp","P1_PP01AD","P2_OnOff","P2_AutoGO","P4_ST_PS")]
        step = n // (attacks + 1)
        for a in range(attacks):
            st = step*(a+1) - 100; dur = int(rng.choice([60, 120, 300, 600])); k = int(rng.choice([1, 2, 3, 6]))
            hit = rng.choice(tags, k, replace=False); mag = rng.choice([1.5, 3, 5])
            for c in hit:
                df.loc[st:st+dur, c] += mag * df[c].std()
            label[st:st+dur+1] = 1
    return df, label

def main(out="data/raw"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for i, nrows in zip(range(1, 5), (30000, 30000, 20000, 20000)):
        df, _ = make(nrows, i, f"2023-01-0{i}", shift=0.05*(i == 4))
        if i == 2:
            df = pd.concat([df, df.iloc[200:210]]); df = df.drop(index=range(5000, 5012))
            df.loc[300:304, "P1_FT01"] = np.nan
        df.to_csv(out / f"hai-train{i}.csv", index=False)
    for i, (nrows, na) in zip((1, 2), ((20000, 5), (30000, 9))):
        df, lab = make(nrows, 10+i, f"2023-02-0{i}", shift=0.08*i, attacks=na)
        df.to_csv(out / f"hai-test{i}.csv", index=False)
        ts = df["timestamp"]
        if i == 2:   # reproduit HAI 23.05 : timestamps de labels tronqués à la minute ("2022-08-17 0:00")
            ts = ts.dt.floor("min").map(lambda t: f"{t.year}-{t.month:02}-{t.day:02} {t.hour}:{t.minute:02}")
        pd.DataFrame({"timestamp": ts, "label": lab}).to_csv(out / f"label-test{i}.csv", index=False)

if __name__ == "__main__":
    main(*sys.argv[1:])
