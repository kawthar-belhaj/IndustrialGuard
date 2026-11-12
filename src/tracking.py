"""MLflow : magasin SQLite local (nécessaire au Model Registry de la Phase 3) + traçabilité code/données."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path


def code_version(root) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "non versionné"


def data_version(root) -> dict:
    """Identité des données : empreinte de la data card + empreintes SHA-256 des fichiers sources qu'elle contient."""
    p = Path(root) / "reports" / "data_card.json"
    if not p.exists():
        return {"data_card_sha256": "absente", "dataset": "inconnu"}
    card = json.loads(p.read_text(encoding="utf-8"))
    return {"data_card_sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16], "dataset": card.get("dataset", "?"),
            "files_sha256": ",".join(f"{k}:{v['sha256'][:8]}" for k, v in card.get("files", {}).items() if v.get("sha256"))}


def init_mlflow(root, experiment: str):
    import mlflow
    root = Path(root)
    mlflow.set_tracking_uri(f"sqlite:///{(root / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment)
    return mlflow
