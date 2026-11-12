import numpy as np, pandas as pd, pytest
from src.data.io import find_hai_files, load_hai_csv, align_labels
from src.data.catalog import segments, parse_summary, crosscheck
from src.data.prep import clean_file, split_frames, prepare_protocol


def _touch(root, rel):
    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("timestamp,A\n"); return p


def _all(root, folder):
    for k in ["hai-train1", "hai-train2", "hai-train3", "hai-train4", "hai-test1", "hai-test2", "label-test1", "label-test2"]:
        _touch(root, f"{folder}/{k}.csv")


def test_no_silent_fallback_to_other_version(tmp_path):
    _all(tmp_path, "hai-22.04")                         # seule une AUTRE version est présente
    with pytest.raises(FileNotFoundError):
        find_hai_files(tmp_path, "23.05")


def test_ambiguity_is_an_error_and_haiend_ignored(tmp_path):
    _all(tmp_path, "hai-23.05"); _all(tmp_path, "haiend-23.05")
    assert len(find_hai_files(tmp_path, "23.05")) == 8  # haiend ignoré
    _touch(tmp_path, "copie/hai-23.05/hai-test1.csv")
    with pytest.raises(FileNotFoundError, match="ambiguës"):
        find_hai_files(tmp_path, "23.05")


def _df(n=180):
    return pd.DataFrame({"timestamp": pd.date_range("2022-01-01 00:00:01", periods=n, freq="1s"), "A": np.arange(n, dtype=float)})


def test_unmatched_or_missing_labels_never_become_normal():
    d = _df(); lab = pd.DataFrame({"timestamp": d.timestamp.iloc[:-5], "label": 0})
    with pytest.raises(ValueError, match="sans label"):
        align_labels(d, lab)
    lab2 = pd.DataFrame({"timestamp": d.timestamp, "label": [0] * (len(d) - 1) + [None]})
    with pytest.raises(ValueError, match="manquants"):
        align_labels(d, lab2)


def test_positional_rejects_unsorted_labels_and_count_mismatch():
    d = _df(); ts = d.timestamp.dt.floor("min").copy(); lab = pd.DataFrame({"timestamp": ts, "label": 0})
    lab.loc[[0, 100], "timestamp"] = lab.loc[[100, 0], "timestamp"].values       # ordre cassé
    with pytest.raises(ValueError):
        align_labels(d, lab)
    lab = pd.DataFrame({"timestamp": ts, "label": 0}); lab.loc[0, "timestamp"] = lab.loc[100, "timestamp"]   # compte par minute faux
    with pytest.raises(ValueError):
        align_labels(d, lab)


def test_clean_file_label_logic():
    d = _df(40); d["label"] = 0; d.loc[10:20, "label"] = 1
    g = d.drop(index=[15, 16, 17])                       # trou DANS une attaque : voisins d'accord -> label 1 conservé
    out = clean_file(g, 1, 5); assert len(out) == 40 and out.loc[out.timestamp == d.timestamp[16], "label"].iloc[0] == 1
    g = d.drop(index=[20, 21])                           # trou à la frontière attaque/normal : ambigu -> lignes supprimées
    out = clean_file(g, 1, 5); assert len(out) == 38
    dd = pd.concat([d, d.iloc[[12]].assign(label=0)])    # doublon avec label contradictoire
    with pytest.raises(ValueError, match="contradictoires"):
        clean_file(dd, 1, 5)


def test_segments_and_crosscheck(tmp_path):
    d = _df(300); d["label"] = 0; d.loc[50:109, "label"] = 1; d.loc[200:230, "label"] = 1
    seg = segments(d); assert list(seg["durée_s"]) == [60, 31]
    f = tmp_path / "s.txt"
    f.write_text("Attacks: 3 times\nTotal: 00:02:00\n[ 1] 2022-01-01 00:00:51 2022-01-01 00:01:50 00:00:59\n[ 2] 2022-01-01 00:03:21 2022-01-01 00:03:51 00:00:30\n")
    summ, head = parse_summary(f); assert head["attacks_header"] == 3 and head["attacks_listed"] == 2
    cc, extra = crosscheck(seg, summ, 2); assert cc["ok"].all() and extra == []
    seg2 = seg.copy(); seg2.loc[0, "début"] += pd.Timedelta(seconds=30)
    assert not crosscheck(seg2, summ, 2)[0]["ok"].all()


def _frames(n=400):
    def mk(start, lab=False, n=n, shift=0.0):
        t = pd.date_range(start, periods=n, freq="1s"); r = np.random.default_rng(1)
        df = pd.DataFrame({"timestamp": t, "A": r.normal(0, 1, n) + shift, "B": r.normal(5, 2, n), "C": 1.0})
        if lab: df["label"] = (np.arange(n) % 100 < 10).astype(int)
        return df
    return {"train1": mk("2022-01-01"), "train2": mk("2022-01-02", n=600), "train3": mk("2022-01-05"), "train4": mk("2022-01-06"),
            "test1": mk("2022-01-03", True), "test2": mk("2022-01-04", True)}


def test_chrono_protocol_orders_and_sessions_flagged():
    c = _frames(); fr = split_frames(c, "chrono", holdout_hours=0.05)
    assert max(fr[k].timestamp.max() for k in ("train", "val_normal", "val")) < fr["test"].timestamp.min() and "stream" in fr
    s = split_frames(c, "sessions"); assert s["train"].timestamp.max() > s["test"].timestamp.min()     # train3 postérieur à test2 : documenté
    c["test1"]["timestamp"] += pd.Timedelta(days=10)                                                    # val APRÈS test -> refus
    with pytest.raises(ValueError):
        split_frames(c, "chrono", holdout_hours=0.05)


def test_unstable_variable_policy_and_no_leakage():
    c = _frames(); c["train4"]["A"] += 500                                                              # variable explosive sur val_normal
    fr = split_frames(c, "sessions")
    keep = prepare_protocol("s", fr, unstable_z=50, unstable_policy="keep_and_flag"); assert "A" in keep.feats and "A" in keep.unstable
    rule = prepare_protocol("s", fr, unstable_z=50, unstable_policy="rule"); assert "A" not in rule.feats and "A" in rule.band_rules
    assert abs(keep.splits["train"][keep.feats].mean().mean()) < 1e-3 and "C" in keep.dropped         # constante retirée, scaler sur le train


def test_embargo_trims_only_contiguous_boundaries_and_session_column():
    from src.data.prep import apply_embargo
    def mk(start, n): return pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="1s"), "A": np.arange(n, dtype=float)})
    c = {"train1": mk("2022-01-01 00:00:00", 2000), "test1": mk("2022-01-02 00:00:00", 2000),
         "train2": mk("2022-01-02 00:33:20", 2000),            # démarre 1 s après la fin de test1 : CONTIGU
         "test2": mk("2022-01-05 00:00:00", 2000), "train3": mk("2022-01-09 00:00:00", 2000), "train4": mk("2022-01-10 00:00:00", 2000)}
    out, log = apply_embargo(c, 300)
    assert len(out["train2"]) == 1700 and len(out["train1"]) == 2000 and len(out["train3"]) == 2000      # seule la frontière contiguë est purgée
    assert [l["fichier"] for l in log] == ["train2"]
    fr = split_frames(out, "sessions"); assert set(fr["train"]["session"]) == {"train1", "train2", "train3"}
    p = prepare_protocol("s", {k: v.assign(label=0) if k in ("val", "test") else v for k, v in fr.items()})
    assert "session" in p.splits["train"].columns and "session" not in p.feats
