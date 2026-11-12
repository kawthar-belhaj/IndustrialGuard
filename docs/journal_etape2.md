# Journal de l'étape 2 — features, fenêtres, détecteurs de référence

## Livré
- `src/features/windows.py` : features causales (niveau, delta, écart-type sur 10/30/60 s), calculées par session, jamais à cheval sur deux fichiers ni sur un trou.
- `src/models/detectors.py` : `maxz` (seuil univarié), `pca` (autoencodeur linéaire), `iforest`.
- `src/eval/events.py` : eTaPR officiel + rappel par attaque, délai de détection, fausses alertes/heure ; trois stratégies de seuil (`q999`, `fa1`, `bestf1`).
- `src/tracking.py` : MLflow en SQLite, version du code et empreinte des données.
- `notebooks/02_features_baselines.ipynb` ; 24 tests automatisés.

## Pièges rencontrés (à ne pas oublier)
1. **eTaPR : convention inversée dans le README** (1/-1) alors que le code lit 0/1 → score parfait par erreur. Détecté en testant la bibliothèque sur un cas jouet avant de l'utiliser.
2. **eTaPR plante si aucune alarme n'est prédite** (`float has no attribute mean`) → garde dans `etapr_scores`.
3. Le dépôt eTaPR n'est pas sur PyPI et sa branche s'appelle `main` : installation par l'URL de la roue (licence MIT).
4. Un trou de temps dans une session la coupe en sous-segments (`train2#1`, `train2#2`) : comportement voulu, le contrôle du notebook l'accepte.

## Statut
Le notebook a été validé sur des données **de démo** (aucune conclusion à en tirer). Les chiffres réels viendront de l'exécution sur HAI 23.05 ; ils seront reportés ici.

## Hypothèses abordées (registre du notebook 1)
H1 (sessions vs chrono), H5 (valeur des features temporelles, fenêtres > 60 s non testées), H6 (stratégies de seuil). Non testées : H2, H3, H4, H7, H8, H9.
