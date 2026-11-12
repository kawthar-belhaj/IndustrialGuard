# Revue du notebook 1 — points soulevés et traitement (v3)

| # | Point de la relecture | Verdict après vérification dans le code | Traitement |
|---|---|---|---|
| 1 | `pick()` : repli silencieux vers une autre version de HAI ; première correspondance choisie sans signalement | **Fondé** | `find_hai_files` : version exacte exigée, erreur si 0 ou plusieurs candidats, `haiend` ignoré ; tests `test_strict.py` |
| 2 | Découpage « temporel » alors que test1 précède train2/train3 | **Fondé** (et test2 précède train3/train4) : l'intitulé était inexact | Renommé « sessions » ; protocole `chrono` ajouté (décisions toutes antérieures au test, train3+4 = flux de production) ; tableau chronologique §4.3 |
| 3 | Concordance à la minute ≠ ordre identique des lignes | **Partiellement fondé** : tri et lignes/minute sont déjà impliqués par l'égalité ligne à ligne, mais l'ordre *à l'intérieur* d'une minute n'est effectivement pas prouvé | Vérifications rendues explicites (tri, lignes/minute, 100 %) **et** recoupement à la seconde avec `summary_label*.txt` (`crosscheck`) |
| 4 | `fillna(0)` : label manquant traité comme normal | **Fondé** (jointure par timestamp, coercition numérique, `clean_file` sur lignes insérées) | Erreur si label manquant/non apparié ; label d'une ligne insérée conservé seulement si les deux voisins concordent ; doublons à labels contradictoires = erreur |
| 5 | Winsorisation : train plafonné, val/test non | **Fondé comme hypothèse** : choix volontaire mais non démontré | Diagnostic (nombre et variables concernées) + hypothèse H2 (ablation en Phase 2) |
| 6 | Constantes : pas une preuve universelle d'attaque | **Fondé** : « règle très fiable » était trop fort | Texte corrigé ; couverture et délai mesurés ; règle présentée comme complément à haute précision, faible couverture |
