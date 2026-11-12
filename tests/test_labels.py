import numpy as np, pandas as pd, pytest
from src.data.io import load_hai_csv


def _data(n=300):
    t = pd.date_range("2022-08-17 00:00:01", periods=n, freq="1s")
    return pd.DataFrame({"timestamp": t, "A": np.arange(n, dtype=float)})


def test_minute_truncated_labels_are_aligned_positionally(tmp_path):
    """Régression HAI test2 : timestamps de labels tronqués à la minute (60 lignes par timestamp)."""
    d = _data(); y = np.zeros(len(d), int); y[100:160] = 1
    d.to_csv(tmp_path / "d.csv", index=False)
    lab = pd.DataFrame({"timestamp": d.timestamp.dt.floor("min").map(lambda t: f"{t.year}-{t.month:02}-{t.day:02} {t.hour}:{t.minute:02}"), "label": y})
    lab.to_csv(tmp_path / "l.csv", index=False)
    info = {}; out = load_hai_csv(tmp_path / "d.csv", tmp_path / "l.csv", info)
    assert len(out) == len(d) and out["label"].sum() == 60
    assert list(info.values())[0]["mode"] == "positionnel"


def test_second_resolution_labels_use_timestamp_join(tmp_path):
    d = _data(); y = np.zeros(len(d), int); y[10:20] = 1
    d.to_csv(tmp_path / "d.csv", index=False); pd.DataFrame({"timestamp": d.timestamp, "label": y}).to_csv(tmp_path / "l.csv", index=False)
    out = load_hai_csv(tmp_path / "d.csv", tmp_path / "l.csv")
    assert out["label"].sum() == 10 and len(out) == len(d)


def test_incompatible_labels_raise(tmp_path):
    d = _data(); d.to_csv(tmp_path / "d.csv", index=False)
    pd.DataFrame({"timestamp": ["2022-08-17 0:00"] * 50, "label": [0] * 50}).to_csv(tmp_path / "l.csv", index=False)
    with pytest.raises(ValueError):
        load_hai_csv(tmp_path / "d.csv", tmp_path / "l.csv")
