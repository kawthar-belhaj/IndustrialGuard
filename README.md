# IndustrialGuard — Phase 1 : Data Engineering

## Démarrage
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git init && dvc init
```

## 1. Récupérer HAI 23.05 depuis Kaggle
https://www.kaggle.com/datasets/icsdataset/hai-security-dataset
- Télécharge le zip (il contient plusieurs versions : ne garde que **hai-23.05**, pas haiend-23.05) et dézippe-le dans `data/raw/`.
- Le notebook retrouve les fichiers tout seul (recherche récursive de `hai-train1..4.csv`, `hai-test1..2.csv`, `label-test1..2.csv`).
- Alternative CLI : `kaggle datasets download -d icsdataset/hai-security-dataset -p data/raw --unzip`

## 1b. Notebook d'étude complet (recommandé)
`notebooks/01_data_engineering.ipynb` : contexte métier, inventaire + SHA-256, dictionnaire des 86 tags, EDA univariée / temporelle / multivariée,
analyse des attaques (durée, signature par variable), dérive naturelle train↔test, qualité, décisions justifiées D1-D11, découpage, data card.
- Figures sauvegardées dans `reports/figures/` (18 PNG, réutilisables dans le rapport).
- Les « 💡 Interprétation » sont **générées à partir des chiffres réels** : elles s'adaptent à tes données.
- Durée indicative sur HAI complet : 5-15 min ; RAM conseillée : ≥ 8 Go.
- Test à blanc sans HAI : `python scripts/make_synthetic_hai_rich.py` (données FICTIVES, ne pas interpréter les résultats), puis supprimer `data/raw/*.csv`.

```bash
pip install jupyterlab matplotlib
jupyter lab        # ouvre notebooks/01_data_engineering.ipynb et exécute les cellules dans l'ordre
```

## 2. Versionner avec DVC
```bash
dvc add data/raw
git add data/raw.dvc .gitignore && git commit -m "data: HAI 23.05 raw (v1)"
git tag data-v1
```

## 3. Lancer la chaîne
```bash
python -m src.data.validate      # -> reports/data_quality.json
python -m src.data.preprocess    # -> data/processed/{sessions,chrono}/ (parquet, scaler, features, stats de référence)
pytest -q                        # 15 tests (labels, sélection des fichiers, protocoles, nettoyage)
# ou, reproductible et versionné :  dvc repro
```
Le **notebook 1** et le **pipeline DVC** importent les mêmes fonctions de `src/data/` : aucune divergence entraînement / production.

## Deux protocoles de découpage (voir notebook §4.3 et `docs/docs_revue_v3.md`)
| Protocole | train | val_normal | val (seuil) | test final | Question |
|---|---|---|---|---|---|
| `sessions` (défaut) | train1+2+3 | train4 | test1 | test2 | généralisation à d'autres sessions (**non chronologique** : train3/4 sont postérieurs à test2) |
| `chrono` | train1 + train2 (sans 10 h) | 10 dernières h de train2 | test1 | test2 | mise en production : tout ce qui décide est antérieur au test ; train3+4 = flux de production simulé |

- Scaler, colonnes retenues et winsorisation sont ajustés sur le **train seul** de chaque protocole.
- Les petits trous (≤ 5 s) sont comblés ; les grands restent des trous. Un label manquant n'est **jamais** converti en « normal ».
- `train_reference_stats.json` = référence du futur Data Drift (Phase 4).

## Leçons apprises sur HAI 23.05 (à ne pas oublier)
- **`label-test2.csv` a des timestamps tronqués à la minute** (`2022-08-17 0:00`, 60 lignes par timestamp) mais une ligne par seconde.
  Une jointure sur le timestamp détruit les labels (taux d'attaque 3,65 % → 0,06 %). Le chargeur aligne donc **par position**, vérifié à la minute près
  (`src/data/io.py::align_labels`, test de régression `tests/test_labels.py`). Des **contrôles bloquants** arrêtent le notebook si le nombre de lignes,
  le nombre d'attaques ou leur taux sont incohérents.
- Le dossier Kaggle contient `hai-23.05/` **et** `haiend-23.05/` : seul `hai-23.05` est utilisé.
- Conseil : garder le projet **hors OneDrive** (ex. `C:\\dev\\industrialguard`) : la synchronisation ralentit les gros CSV, verrouille des fichiers et perturbe DVC/Git.
- `STRICT_OFFICIAL_CHECKS = False` (cellule de configuration) uniquement pour un test à blanc sur les données de démo.

## Versionner l'étape 1 (Windows PowerShell, à exécuter UNE fois, projet hors OneDrive)
```powershell
git init; dvc init
Remove-Item -Recurse -Force data\processed\_obsolete      # anciens fichiers (v1) avec labels faux : ne pas les versionner
git add .; git commit -m "chore: IndustrialGuard phase 1 (code, tests, notebook)"
dvc add data\raw data\processed
git add data\raw.dvc data\processed.dvc data\.gitignore reports\data_card.json reports\data_quality.json
git commit -m "data: HAI 23.05 v1 (raw + processed sessions/chrono)"
git tag data-v1
```
Ensuite, relancer l'en-tête du notebook : `git_commit` doit afficher un identifiant au lieu de « non versionné ».

## Phase 2 — notebook 2 : features, fenêtres, détecteurs de référence, MLflow
```powershell
pip install -r requirements.txt          # installe aussi MLflow et la métrique officielle eTaPR
jupyter lab                              # notebooks/02_features_baselines.ipynb (après le notebook 1)
mlflow ui --backend-store-uri sqlite:///mlflow.db     # explorer les runs (depuis la racine du projet)
```
- **Convention eTaPR : 0 = normal, 1 = anomalie.** Le README du dépôt eTaPR annonce « 1 / -1 » mais son code lit 0 / 1 : avec l'encodage du README le score est parfait *par erreur* (démontré dans le notebook, testé dans `tests/test_features_eval.py`). La bibliothèque plante aussi si aucune alarme n'est émise : `src/eval/events.py` protège ce cas.
- **Fenêtres causales par session** (`src/features/windows.py`) : aucune feature n'utilise le futur, aucune fenêtre ne traverse deux fichiers ni un trou de temps.
- **MLflow en SQLite** (`mlflow.db`, `mlartifacts/`) : indispensable au Model Registry de la Phase 3. Ces deux éléments ne sont pas versionnés dans Git.
- Aucun réglage n'est fait sur le test final : le seuil se choisit sur `val_normal` et `val`.

## EDA en 9 parties (plan de travail) — avancement
| Partie | Notebook | État |
|---|---|---|
| 1-2 Dataset Understanding + Data Quality | `01_Dataset_Understanding1.ipynb` (ton travail) | fait |
| 3 Univarié | — | à faire |
| **4 Target Variable Analysis** (points 27 à 33) | `notebooks/04_Target_Variable_Analysis.ipynb` | **livré** |
| 5 Multivarié et features | — | à faire |
| 6 Comportement temporel | — | à faire |
| **7 Temporal Data Drift** (points 53 à 64) | `notebooks/07_Temporal_Data_Drift_Analysis.ipynb` | **livré** |
| 8 Industriel structurel et fonctionnel | — | à faire (nécessite `hai_dataset_technical_details.pdf`) |
| 9 Synthèse et recommandations | — | à faire en dernier |

Briques réutilisables et testées : `src/eda/target.py` (transitions, séquences, déséquilibre), `src/eda/drift.py` (PSI, KS, Wasserstein normalisée, JS, fenêtres, seuil nul), `src/eda/tools.py` (typologie, AUC univariée, ACF, spectre, stationnarité).

**Règles d'usage des données** (communes à toutes les parties) : référence et seuils de drift calculés sur le **train** ; labels utilisés de façon **descriptive** ; décisions fondées sur les labels avec **test1 seulement**, test2 en lecture seule.
