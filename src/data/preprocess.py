"""Pipeline de prétraitement (DVC) : mêmes fonctions que le notebook 1 -> aucune divergence entraînement/production."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from src.data.io import find_hai_files, load_hai_csv
from src.data.prep import clean_file, apply_embargo, split_frames, prepare_protocol, save_prepared


def run(params_path: str = "params.yaml") -> dict:
    p = yaml.safe_load(open(params_path)); d, pp = p["data"], p["preprocess"]
    f = find_hai_files(d["raw_dir"], d.get("version_tag", "23.05"), d.get("allow_unversioned", False))
    info = {}
    raw = {f"train{i}": load_hai_csv(f[f"hai-train{i}"]) for i in range(1, 5)}
    raw.update({f"test{i}": load_hai_csv(f[f"hai-test{i}"], f[f"label-test{i}"], info) for i in (1, 2)})
    clean = {k: clean_file(v, d["freq_seconds"], pp["max_ffill_seconds"]) for k, v in raw.items()}
    clean, embargo_log = apply_embargo(clean, pp.get("embargo_seconds", 0))
    out = Path(d["processed_dir"]); metas = {}
    for proto in pp["protocols"]:
        frames = split_frames(clean, proto, pp["holdout_hours"], pp.get("embargo_seconds", 0))
        prep = prepare_protocol(proto, frames, pp["train_clip_sigma"], pp["exclude_columns"], pp["unstable_z"], pp["unstable_policy"])
        metas[proto] = save_prepared(prep, out / proto)
    summary = {"label_alignment": info, "embargo": embargo_log, "protocols": metas}
    json.dump(summary, open(out / "meta.json", "w"), indent=2, default=str)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--params", default="params.yaml")
    print(json.dumps(run(ap.parse_args().params), indent=2, default=str)[:3000])
