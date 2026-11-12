# Journal de l'étape 1 — faits établis, points ouverts

## Établi sur les vraies données HAI 23.05 (run du notebook v3)
- 6 fichiers de données + 2 fichiers de labels, version `hai-23.05` exacte ; 86 capteurs ; durées identiques à la documentation officielle.
- 0 NaN, 0 doublon, 0 trou de timestamp : le nettoyage retire/insère 0 ligne.
- Labels : `label-test1` (jointure timestamp) et `label-test2` (alignement positionnel, timestamps tronqués à la minute).
  **Recoupement avec `summary_label*.txt` : 14/14 et 38/38 attaques, écart maximal début/fin = 0,0 s ; durées = listées + 1 s/attaque.**
- `summary_label2.txt` : l'en-tête annonce 40 attaques / 8 735 s, la liste en contient 38 / 8 365 s. Les labels suivent la liste.
- Chronologie réelle : train1 < test1 < train2 < test2 < train3 < train4 ; `test1→train2` et `test2→train3` sont **contigus (1 s)**.
- 20 variables constantes sur le train ; `P2_Emerg` et `P2_OnOff` ne changent que pendant 2 attaques sur 52 (délai médian 27 s).
- Dimension intrinsèque : 25 composantes pour 95 % de variance. Autocorrélation médiane 0,91 à 60 s, < 0,5 vers 600 s.
- Drift : 45 % des variables « dérivent » (KS) entre sessions normales, mais le décalage médian n'est que de 0,15–0,31 σ.
- Quasi-constantes à événements rares (σ ≤ 0,05) : `P1_PCV02D` (|z| = 93 sur données normales, 12,0 au repos), `P4_HT_FD` (73 % des valeurs écrêtées), `P4_ST_FD`, `P2_24Vdc`.

## Résultats du placebo (run v3.1 sur les vraies données)
- Seuils **bruts** : 98,8 % des fenêtres NORMALES (260 fenêtres, mêmes durées que les 52 attaques) ont ≥ 1 variable « affectée » (médiane 5, max 25), contre 100 % des attaques (médiane 8,5).
  Conclusion : « toutes les attaques ont une variable affectée » **ne prouve rien**. Mon résultat « 52/52 » de la v3 est à ne pas citer.
- 22 variables sont « affectées » dans plus de 10 % des fenêtres normales ; `P1_TIT03` dans 73 % (variable à tendance lente) : elle n'est pas une signature d'attaque.
- Seuils **calibrés par variable** (99e centile du placebo) : 46/52 attaques ont ≥ 1 variable significative, 6 aucune, médiane 3 variables, 13 attaques n'en affectent qu'une seule.
  Variables les plus spécifiques : `P1_PCV01D` (+40 pts), `P1_PIT01` (+38), `P1_LCV01D` (+31), `P1_LIT01` (+25), `P1_LCV01Z` (+24).
- Limite corrigée en v3.2 : avec 66 seuils simultanés, le « 46/52 » n'a de sens qu'à côté du taux de bruit sur fenêtres normales avec des seuils appris ailleurs.
  **Résultat (validation croisée, run v3.2 sur les vraies données)** : 44 % des fenêtres normales dépassent au moins un seuil, contre 90 % des attaques ;
  médiane de variables au-dessus du seuil : 0 en placebo, 3,5 en attaque ; **AUC = 0,80 [IC95 % bootstrap 0,74–0,86]** ; 38 % des attaques détectées à ≈ 5 % de fausses alertes.
  C'est une borne **optimiste** (durée de chaque attaque connue) et la **référence univariée à battre en Phase 2**.

## Corrigé après analyse (v3.1 → v3.2)
| Constat | Correction |
|---|---|
| Signature « 52/52 attaques affectées, même à 12 σ » : mesure sans groupe témoin (valeurs identiques à 104,4 σ = artefact de plancher) | Placebo de fenêtres normales de même durée ; seuils calibrés par variable (99e centile) ; variables non spécifiques listées |
| Fichiers périmés à la racine de `data/processed` (issus des versions 1 et 2, dont des labels faux) | Déplacés automatiquement vers `data/processed/_obsolete/` |
| Frontières contiguës entre sessions (état du procédé prolongé) | Purge de 600 s côté apprentissage (`apply_embargo`) |
| Fenêtres temporelles pouvant enjamber deux fichiers concaténés | Colonne `session` conservée ; fenêtrage par session obligatoire en Phase 2 |
| `P1_PCV02D` : message « saut ponctuel » inexact | Détection des variables à événements rares : 99,96 % des points valent 12,0 au repos (famille 1) ; `P4_HT_FD`, `P4_ST_FD`, `P2_24Vdc` forment une famille 2 à σ de l'ordre du millième (73 % des valeurs écrêtées pour `P4_HT_FD`) |
| Seuils calibrés sans contrôle des tests multiples (66 variables) | Validation croisée du placebo + AUC bootstrap (v3.2) |

## Points ouverts (hypothèses à trancher en Phase 2)
H1 sessions vs chrono · H2 winsorisation · H3 scaler robuste · H4 variables quasi constantes · H5 fenêtres (10/30/60 + 120/300 s) ·
H6 choix du seuil (val = test1, 14 attaques, session la plus atypique) · H7 variables corrélées · H8 règles + modèle · H9 purge aux frontières.

## Clôture de l'étape 1
Critères satisfaits : chargement strict (version exacte), labels recoupés à 0,0 s des 52 attaques officielles, 25+ contrôles bloquants au vert, deux protocoles produits (`sessions`, `chrono`),
purge des frontières contiguës, colonne `session`, artefacts périmés isolés, data card et catalogue d'attaques exportés, 15 tests automatisés.
**Condition restante : versionnement Git + DVC** (le notebook affiche `git_commit : non versionné`). Tant que `data-v1` n'existe pas, l'étape n'est pas reproductible.
