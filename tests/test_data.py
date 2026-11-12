import json
import numpy as np, pandas as pd, yaml
from scripts.make_synthetic_hai import main as make_data
from src.data import validate, preprocess
from src.data.io import load_hai_csv
from src.data.prep import clean_file


def _setup(tmp_path):
    raw = tmp_path / "raw"; make_data(str(raw))
    p = yaml.safe_load(open("params.yaml"))
    p["data"].update(raw_dir=str(raw), processed_dir=str(tmp_path / "proc"), report_path=str(tmp_path / "rep.json"), allow_unversioned=True)
    p["preprocess"]["holdout_hours"] = 0.3
    path = tmp_path / "params.yaml"; yaml.safe_dump(p, open(path, "w")); return str(path), tmp_path / "proc"


def test_validation_detects_injected_problems(tmp_path):
    params, _ = _setup(tmp_path); f = validate.run(params)["files"][0]
    assert f["duplicate_timestamps"] == 5 and f["missing_cells"] == 6
    assert f["time_gaps"]["count"] == 1 and f["time_gaps"]["missing_seconds"] == 20
    assert "P4_ST_PS" in f["constant_columns"]


def test_clean_file_never_invents_long_gaps(tmp_path):
    _setup(tmp_path); df = clean_file(load_hai_csv(tmp_path / "raw" / "hai-train1.csv"), 1, 5)
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing and df.isna().sum().sum() == 0
    assert len(df) == 4000 - 15 - 1      # trou de 20 s : 5 comblées, 15 supprimées ; 6 NaN consécutifs : 5 comblés, 1 supprimée


def test_both_protocols_no_leakage_and_chrono_order(tmp_path):
    params, proc = _setup(tmp_path); preprocess.run(params)
    for proto in ("sessions", "chrono"):
        train = pd.read_parquet(proc / proto / "train.parquet"); feats = json.load(open(proc / proto / "features.json"))
        assert "P4_ST_PS" not in feats and len(feats) == 4
        assert np.allclose(train[feats].mean(), 0, atol=.15)                      # scaler ajusté sur le train
    ch = {k: pd.read_parquet(proc / "chrono" / f"{k}.parquet") for k in ("train", "val_normal", "val", "test")}
    assert max(ch[k]["timestamp"].max() for k in ("train", "val_normal", "val")) < ch["test"]["timestamp"].min()
    assert (proc / "chrono" / "stream_normal.parquet").exists() and set(ch["test"]["label"]) == {0, 1}
