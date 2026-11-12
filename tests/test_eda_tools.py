import numpy as np, pandas as pd, pytest
from src.eda import tools as T
from src.features.windows import build_features, feature_columns


def test_tag_info_heuristics():
    assert T.tag_info("P1_FCV01D")["rôle_probable"] == "commande (demande)" and T.tag_info("P1_FCV01Z")["rôle_probable"] == "retour de position"
    assert T.tag_info("P1_PIT01")["rôle_probable"] == "mesure (capteur)" and T.tag_info("x1002_08_SETPOINT_OUT")["processus"].startswith("Contrôleurs")
    assert T.tag_info("P1_STSP")["rôle_probable"] == "consigne"


def test_auc_per_column_detects_signal_and_ignores_noise():
    rng = np.random.default_rng(0); y = np.r_[np.zeros(500), np.ones(200)]
    X = pd.DataFrame({"signal": np.r_[rng.normal(0, 1, 500), rng.normal(3, 1, 200)], "bruit": rng.normal(0, 1, 700), "inverse": np.r_[rng.normal(3, 1, 500), rng.normal(0, 1, 200)]})
    a = T.auc_per_column(X, y)
    assert a["signal"] > .95 and abs(a["bruit"] - .5) < .08 and a["inverse"] < .05                 # signée : le sens de l'effet est conservé


def test_periodogram_finds_period_and_acf_half_life():
    t = np.arange(4000); x = np.sin(2 * np.pi * t / 100) + .1 * np.random.default_rng(1).normal(size=len(t))
    assert abs(T.periodogram_peaks(x, 1.0, 1)[0][0] - 100) < 3
    ar = np.zeros(20000); e = np.random.default_rng(2).normal(size=20000)
    for i in range(1, 20000): ar[i] = .99 * ar[i - 1] + e[i]
    assert 30 < T.acf_half_life(ar) < 120                                                             # 0,99^k = 0,5  =>  k ~ 69


def test_stationarity_verdicts():
    rng = np.random.default_rng(3); assert T.stationarity(rng.normal(size=3000))["verdict"] == "stationnaire"
    assert T.stationarity(np.cumsum(rng.normal(size=3000)))["verdict"] in ("racine unitaire / tendance", "non concluant")


def test_bimodality_and_rest_share():
    rng = np.random.default_rng(4); bi = np.r_[rng.normal(-3, 1, 2000), rng.normal(3, 1, 2000)]
    assert T.bimodality_coefficient(bi) > .555 > T.bimodality_coefficient(rng.normal(size=4000)); assert T.rest_share(np.r_[np.full(95, 12.0), np.arange(5.)]) >= .95


def _df(n=600):
    rng = np.random.default_rng(5); d = pd.DataFrame({"timestamp": pd.date_range("2022-01-01", periods=n, freq="1s"), "x": rng.normal(size=n).cumsum(), "label": 0})
    d["session"] = "A"; return d


def test_new_feature_families_values_and_causality():
    df = _df(); fs = build_features(df, ["x"], (10, 30), 5, ("level", "dev", "range"))
    assert list(fs.X.columns) == feature_columns(["x"], (10, 30), ("level", "dev", "range"))
    pos = int(fs.meta.pos.iloc[3]); w = df.x.to_numpy()[pos - 9:pos + 1]
    assert np.isclose(fs.X["x|dev10"].iloc[3], df.x.iloc[pos] - w.mean(), atol=1e-4) and np.isclose(fs.X["x|rng10"].iloc[3], w.max() - w.min(), atol=1e-4)
    df2 = df.copy(); df2.loc[df2.index >= 300, "x"] += 1e3; fs2 = build_features(df2, ["x"], (10, 30), 5, ("level", "dev", "range"))
    past = fs.meta.pos < 300; assert np.allclose(fs.X[past].to_numpy(), fs2.X[past].to_numpy())
