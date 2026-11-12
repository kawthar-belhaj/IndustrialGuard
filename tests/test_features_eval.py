import numpy as np, pandas as pd, pytest
from src.features.windows import build_features, feature_columns
from src.models.detectors import make_detector
from src.eval import events as ev


def _df(n=400, sessions=(("A", "2022-01-01 00:00:00"), ("B", "2022-03-01 00:00:00")), seed=0):
    rng = np.random.default_rng(seed); parts = []
    for s, t0 in sessions:
        d = pd.DataFrame({"timestamp": pd.date_range(t0, periods=n, freq="1s"), "x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n)})
        d["label"] = 0; d["session"] = s; parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_features_are_causal():
    df = _df(); a = build_features(df, ["x", "y"], (10, 30), 5); df2 = df.copy(); cut = 200
    df2.loc[(df2.session == "A") & (df2.index >= cut), ["x", "y"]] += 1000                      # on modifie le FUTUR de la session A
    b = build_features(df2, ["x", "y"], (10, 30), 5); keep = (a.meta.session == "A") & (a.meta.pos < cut)
    assert np.allclose(a.X[keep].to_numpy(), b.X[keep].to_numpy())                                # le passé ne bouge pas


def test_no_leakage_across_sessions():
    df = _df(); a = build_features(df, ["x", "y"], (10, 30), 5); df2 = df.copy(); df2.loc[df2.session == "A", ["x", "y"]] += 1e6
    b = build_features(df2, ["x", "y"], (10, 30), 5); mb = a.meta.session == "B"
    assert np.allclose(a.X[mb].to_numpy(), b.X[mb].to_numpy())                                    # la session B ignore totalement A
    assert a.meta[a.meta.session == "B"].pos.min() == 30                                          # fenêtre incomplète écartée au début de chaque session


def test_time_gap_splits_segments_and_names():
    df = _df(300, sessions=(("A", "2022-01-01 00:00:00"),)); df = df.drop(index=range(100, 110)).reset_index(drop=True)
    fs = build_features(df, ["x"], (10,), 5)
    assert set(fs.y_streams) == {"A#1", "A#2"}                                                     # aucune fenêtre ne traverse le trou
    assert list(fs.X.columns) == feature_columns(["x"], (10,), ("level", "delta", "std"))


def test_etapr_convention_and_empty_prediction():
    pytest.importorskip("eTaPR_pkg")
    y = np.zeros(200, int); y[50:80] = 1; y[120:150] = 1
    assert ev.etapr_scores(y, y)["etapr_f1"] == pytest.approx(1.0)                                 # parfait = 1 (0 normal / 1 anomalie)
    assert ev.etapr_scores(y, np.zeros(200, int))["etapr_f1"] == 0.0                               # prédiction vide : pas de plantage
    p = np.zeros(200, int); p[60:70] = 1; p[5:8] = 1
    assert 0 < ev.etapr_scores(y, p)["etapr_f1"] < 0.6                                             # partiel + fausse alarme : score intermédiaire


def test_event_report_counts_delay_and_false_alarms():
    y = np.zeros(7200, int); y[1000:1100] = 1; y[3000:3100] = 1; p = np.zeros(7200, int)
    p[1040:1050] = 1                                                                               # détecte la 1re après 40 s
    p[5000:5010] = 1                                                                               # fausse alarme ; la 2e attaque est manquée
    r = ev.event_report(y, p, grace_s=60)
    assert (r["n_attacks"], r["n_detected"], r["n_false_alarm_segments"]) == (2, 1, 1)
    assert [a["detected"] for a in r["attacks"]] == [True, False] and r["attacks"][0]["delay_s"] == 40 and r["attacks"][1]["delay_s"] is None
    assert r["median_delay_s"] == 40 and r["fa_per_hour"] == pytest.approx(1 / ((7200 - 200) / 3600), rel=1e-3)


def test_fa_budget_threshold_respects_budget():
    pytest.importorskip("eTaPR_pkg")
    df = _df(3600 * 3, sessions=(("N", "2022-01-01 00:00:00"),)); fs = build_features(df, ["x", "y"], (10,), 10, ("level", "std"))
    sc = ev.smooth_scores(make_detector("pca").fit(fs.X).score(fs.X), fs.meta.session, 3)
    thr = ev.thr_fa_budget(fs, sc, budget_per_hour=2.0); y, p = ev.concat_streams(fs, ev.flags_to_streams(fs, sc > thr))
    assert len(ev.runs(p)) / (len(y) / 3600) <= 2.0 + 1e-9


@pytest.mark.parametrize("name", ["maxz", "pca", "iforest"])
def test_detectors_rank_outliers_higher(name):
    rng = np.random.default_rng(1); Xn = rng.normal(0, 1, (2000, 6)); Xa = rng.normal(0, 1, (50, 6)); Xa[:, :2] += 8
    from sklearn.metrics import roc_auc_score
    d = make_detector(name).fit(Xn); Xt = rng.normal(0, 1, (500, 6))
    sc = np.r_[d.score(Xt), d.score(Xa)]; lab = np.r_[np.zeros(500), np.ones(50)]
    assert roc_auc_score(lab, sc) > 0.95                                                           # critère de RANG : les échelles de score diffèrent selon le modèle
