import numpy as np, pandas as pd, pytest
from src.eda import target as TG, drift as DR


def test_transition_matrix_and_runs():
    l = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    P = TG.transition_matrix(l); assert np.allclose(P.sum(axis=1), 1) and np.isclose(P[1, 1], 2 / 4) and np.isclose(P[0, 1], 2 / 5)
    assert list(TG.run_lengths(l, 1)) == [3, 1] and list(TG.run_lengths(l, 0)) == [2, 3, 1]


def test_coverage_cv_and_trivial_classifiers():
    assert TG.duration_coverage([5, 20, 70, 200], (10, 60)) == {10: .75, 60: .5}
    assert TG.interarrival_cv([0, 100, 200, 300]) == pytest.approx(0.0)
    t = TG.trivial_classifiers(.04); assert t["toujours « normal »"]["accuracy"] == pytest.approx(.96) and t["toujours « normal »"]["F1"] == 0.0


def test_psi_js_wd_ks_behave():
    rng = np.random.default_rng(0); ref = rng.normal(0, 1, 20000); same = rng.normal(0, 1, 3000); shifted = rng.normal(1, 1, 3000)
    R = DR.RefDist(ref)
    assert R.psi(same) < .05 < .25 < R.psi(shifted) and R.js(same) < .1 < R.js(shifted)
    assert R.wd(shifted) == pytest.approx(1.0, abs=.1) and R.wd(same) < .1
    assert R.ks(same)[0] < R.ks(shifted)[0]


def test_discrete_variable_and_unseen_category():
    rng = np.random.default_rng(1); ref = rng.choice([0, 1], 5000, p=[.9, .1]); R = DR.RefDist(ref)
    assert R.psi(rng.choice([0, 1], 800, p=[.9, .1])) < .05 and R.psi(rng.choice([0, 1], 800, p=[.5, .5])) > .25
    assert R.psi(np.full(500, 7.0)) > .25                                   # valeur jamais vue dans la référence : signalée


def test_window_slices_never_overfill_and_drop_short_tail():
    assert DR.window_slices(10000, 3600) == [(0, 3600), (3600, 7200)]       # queue de 2 800 s < 90 % : écartée
    assert DR.window_slices(10900, 3600)[-1] == (7200, 10800)


def test_null_thresholds_and_compute_drift_shapes():
    rng = np.random.default_rng(2); X = rng.normal(size=(8000, 2)); cols = ["a", "b"]; refs = {c: DR.RefDist(X[:, j]) for j, c in enumerate(cols)}
    wins = [{"window": f"w{i}", "session": "s", "start": pd.Timestamp("2022-01-01") + pd.Timedelta(hours=i), "data": X[i * 2000:(i + 1) * 2000], "attack_frac": 0.0} for i in range(4)]
    d = DR.compute_drift(refs, wins, cols); assert len(d) == 8 and {"psi", "ks", "wd", "js"} <= set(d.columns)
    th = DR.null_thresholds(d); assert list(th.index) == cols and (th["psi"] >= 0).all()
    assert DR.local_wd(X[:500], X[:500] + 2.0, np.array([1.0, 1.0])) == pytest.approx([2.0, 2.0], abs=.05)
