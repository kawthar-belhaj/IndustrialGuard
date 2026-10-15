# 🛡️ IndustrialGuard

### Plateforme MLOps intelligente pour la détection d’anomalies industrielles et la gestion automatisée du cycle de vie des modèles

**Transformer les données industrielles en services intelligents grâce à la détection d’anomalies, au monitoring continu et à une maintenance contrôlée des modèles de Machine Learning.**

---

## Sommaire

- [1. En bref](#1-en-bref)
- [2. Problématique](#2-problématique)
- [3. Objectifs du projet](#3-objectifs-du-projet)
- [4. Jeu de données](#4-jeu-de-données)
- [5. Fonctionnalités principales](#5-fonctionnalités-principales)
- [6. Architecture](#6-architecture)
- [7. Technologies utilisées](#7-technologies-utilisées)
- [8. Approches de Machine Learning](#8-approches-de-machine-learning)
- [9. Cycle de vie MLOps](#9-cycle-de-vie-mlops)
- [10. Structure du dépôt](#10-structure-du-dépôt)
- [11. Installation et démarrage](#11-installation-et-démarrage)
- [12. Feuille de route](#12-feuille-de-route)
- [13. Limites et utilisation responsable](#13-limites-et-utilisation-responsable)

---

## 1. En bref

IndustrialGuard est une plateforme MLOps intelligente dédiée à la détection d’anomalies dans les systèmes industriels et à la gestion complète du cycle de vie des modèles de Machine Learning.

Les environnements industriels, tels que les chaudières, les turbines et les systèmes de traitement de l’eau, génèrent continuellement des séries temporelles provenant de capteurs et d’actionneurs. Les variations inhabituelles de ces signaux peuvent révéler des anomalies de fonctionnement, des défaillances potentielles ou des activités malveillantes.

Cependant, développer un modèle capable de détecter des anomalies ne suffit pas. Il faut également pouvoir le déployer, surveiller son comportement, évaluer ses performances et le maintenir lorsque les conditions opérationnelles évoluent.

IndustrialGuard vise à répondre à ce besoin en intégrant la détection d’anomalies, le suivi des données et des modèles, l’analyse de leur état de santé et la prise de décision concernant le réentraînement et le déploiement de nouvelles versions.

La plateforme suit un cycle de vie complet :

**Données industrielles → Prétraitement → Entraînement → Validation → Déploiement → Monitoring → Prise de décision → Réentraînement → Nouvelle validation**

L’objectif est de transformer un modèle expérimental en un service de Machine Learning traçable, reproductible et maintenable.

## 2. Problématique

Les systèmes de détection d’anomalies industrielles présentent plusieurs défis :

- **Des données complexes et imparfaites :** les mesures issues des capteurs peuvent contenir des valeurs manquantes, des doublons, des incohérences ou des observations anormales.
- **L’évolution des conditions industrielles :** les changements dans la distribution des données peuvent affecter la fiabilité des modèles déployés.
- **Un manque de visibilité sur les modèles :** sans surveillance adaptée, une dégradation des performances peut passer inaperçue.
- **Des réentraînements potentiellement inutiles :** une dérive des données ne signifie pas systématiquement que le modèle doit être réentraîné.
- **Des risques liés au déploiement :** remplacer un modèle en production sans validation suffisante peut introduire de nouvelles erreurs.
- **Un besoin de traçabilité :** les expériences, les versions des jeux de données, les configurations des modèles et les décisions de déploiement doivent être enregistrées pour garantir la reproductibilité.

IndustrialGuard propose une chaîne MLOps unifiée qui relie la détection d’anomalies, le monitoring, la gouvernance des modèles et la gestion contrôlée de leur cycle de vie.

## 3. Objectifs du projet

L’objectif général est de développer une plateforme capable de gérer le cycle de vie des modèles de détection d’anomalies industrielles, de la préparation des données jusqu’au monitoring en production et à la maintenance des modèles.

Les objectifs spécifiques sont les suivants :

1. **Préparer des données fiables** grâce à la validation, au prétraitement et à la création de variables temporelles.
2. **Détecter les anomalies** en entraînant et en comparant différentes approches de Machine Learning et de Deep Learning.
3. **Assurer la traçabilité des expériences et des modèles** grâce au versionnement des données, au suivi des expériences et au registre des modèles.
4. **Exposer les prédictions via une API** afin de rendre les fonctionnalités de détection accessibles à d’autres applications.
5. **Surveiller les données, les modèles et l’infrastructure** pour identifier les dérives, les dégradations de performances et les problèmes opérationnels.
6. **Prendre des décisions de réentraînement adaptées** à partir d’un moteur de décision analysant plusieurs indicateurs de monitoring.
7. **Valider les nouveaux modèles** avant leur promotion en production.
8. **Sécuriser le déploiement** grâce à la gestion des versions, au déploiement progressif et aux mécanismes de retour à une version stable.

Le principe central d’IndustrialGuard est le suivant : **une dérive détectée doit déclencher une analyse, et non systématiquement un réentraînement automatique.** Toute nouvelle version doit respecter des critères de validation avant son déploiement.
## 4. Jeu de données

### 4.1 Présentation du dataset

IndustrialGuard s'appuie sur le **HAI (*HIL-based Augmented ICS Security Dataset*)**, un jeu de données public conçu pour l'étude de la sécurité et de la détection d'anomalies dans les systèmes de contrôle industriels (*Industrial Control Systems — ICS*).

Il est issu d'un environnement industriel expérimental combinant plusieurs processus physiques et un simulateur *Hardware-in-the-Loop (HIL)*. Cet environnement reproduit notamment des processus de production d'énergie et des interactions entre différents composants industriels.

Le dataset contient des séries temporelles issues de capteurs et de variables de supervision, enregistrées dans des conditions normales et lors de scénarios d'attaque. Il permet ainsi d'étudier les comportements habituels des systèmes industriels et les variations associées à des événements anormaux.

**Sources officielles :**

- [Dépôt GitHub — HAI Security Dataset](https://github.com/icsdataset/hai)
- [Téléchargement sur Kaggle — HAI Security Dataset](https://www.kaggle.com/datasets/icsdataset/hai-security-dataset)

### 4.2 Caractéristiques du dataset

IndustrialGuard privilégiera la version **HAI 23.05**, sous réserve de confirmation de la version effectivement téléchargée et utilisée dans le pipeline.

| Caractéristique | Description |
|---|---|
| Nom complet | HIL-based Augmented ICS Security Dataset |
| Version ciblée | HAI 23.05 |
| Type de données | Séries temporelles industrielles |
| Format principal | Fichiers CSV |
| Sources des mesures | Capteurs et variables de supervision industrielle |
| Fréquence des données | 86 points de données par seconde pour HAI 23.05 |
| Données d'entraînement | Fichiers contenant des périodes de fonctionnement normal |
| Données de test | Fichiers contenant des périodes normales et des scénarios d'attaque |
| Utilisation | Détection d'anomalies, évaluation temporelle et expérimentation MLOps |

*Source : documentation officielle du dataset HAI.*

### 4.3 Organisation des données HAI 23.05

La version HAI 23.05 comprend plusieurs fichiers d'entraînement et de test.

| Type | Fichiers | Durée cumulée indiquée | Volume approximatif |
|---|---|---:|---:|
| Entraînement | `hai-train1.csv` à `hai-train4.csv` | 249 heures | 494,8 MB |
| Test | `hai-test1.csv` et `hai-test2.csv` | 79 heures | 156,6 MB |

Les fichiers d'entraînement couvrent des périodes de fonctionnement normal. Les fichiers de test comprennent des périodes normales et des attaques destinées à évaluer les capacités de détection des modèles.

Les fichiers CSV contiennent une colonne temporelle, des variables industrielles et des étiquettes permettant d'identifier les attaques. La structure exacte des colonnes dépend de la version du dataset.

### 4.4 Exploitation des données dans IndustrialGuard

Le dataset HAI constituera la base des expérimentations de détection d'anomalies et de validation de la chaîne MLOps.

Son exploitation suivra plusieurs étapes :

1. **Acquisition et organisation** des fichiers du dataset.
2. **Versionnement des données** avec DVC afin de conserver la traçabilité des versions utilisées.
3. **Analyse exploratoire** des variables industrielles, des séries temporelles et des étiquettes d'attaque.
4. **Validation et prétraitement** : contrôle des valeurs manquantes, des doublons, des incohérences et de la continuité temporelle.
5. **Feature Engineering temporel** : calcul de statistiques, de variations et de caractéristiques sur des fenêtres temporelles.
6. **Entraînement et évaluation** des modèles de détection d'anomalies.
7. **Suivi des expériences** avec MLflow pour relier les résultats aux données, aux configurations et aux versions des modèles.

Les séparations entre apprentissage, validation et test respecteront l'ordre temporel afin de limiter les fuites de données. Les étiquettes d'attaque seront utilisées pour l'évaluation lorsque disponibles, sans les intégrer par erreur aux variables d'entrée du modèle.

### 4.5 Pourquoi le dataset HAI ?

Le choix du dataset HAI répond à plusieurs besoins du projet :

- **Un contexte industriel réaliste :** les données proviennent d'un banc d'essai de systèmes de contrôle industriels.
- **Une dimension temporelle :** les observations successives permettent d'étudier les changements de comportement.
- **Des scénarios d'attaque documentés :** ils permettent d'évaluer la détection sur des événements identifiés.
- **Une base pour comparer les modèles :** les mêmes données peuvent servir à comparer Isolation Forest, Autoencoder et LSTM Autoencoder.
- **Une application concrète du MLOps :** les données peuvent servir à expérimenter le versionnement, le suivi des modèles, le monitoring et les mécanismes contrôlés de réentraînement.

**Remarque :** les résultats de détection, les métriques de performance et les conclusions expérimentales seront ajoutés au README après l'exécution effective des expériences.
## 5. Fonctionnalités principales

IndustrialGuard est conçu selon une approche progressive : un MVP permettant de construire et de déployer une première chaîne de détection d’anomalies, puis des évolutions visant à renforcer le monitoring, l’automatisation et la fiabilité du cycle de vie des modèles.

### 5.1 Périmètre du MVP

Le MVP se concentre sur les fonctionnalités essentielles à la mise en place d’une première plateforme MLOps.

- **Préparation des données industrielles :** nettoyage, validation et prétraitement des séries temporelles issues du dataset HAI 23.05.
- **Feature Engineering temporel :** création de variables pertinentes à partir des mesures industrielles et de fenêtres temporelles.
- **Détection d’anomalies :** mise en place d’un modèle de référence basé sur Isolation Forest, avec la possibilité d’évaluer une approche Autoencoder.
- **Suivi des expérimentations :** utilisation de MLflow pour enregistrer les paramètres, les métriques et les versions des modèles.
- **Gestion des modèles :** organisation et suivi des modèles candidats à l’aide du Model Registry de MLflow.
- **API de prédiction :** exposition du modèle de détection à travers une API développée avec FastAPI.
- **Conteneurisation :** utilisation de Docker pour faciliter le déploiement et la reproductibilité de l’environnement.
- **Monitoring initial :** intégration de Prometheus et Grafana pour suivre les métriques techniques de l’application.

### 5.2 Évolutions futures

Après la validation du MVP, IndustrialGuard pourra évoluer vers une plateforme plus complète intégrant les fonctionnalités suivantes.

**Détection avancée et explicabilité**
- Évaluation de modèles complémentaires, notamment les Autoencoders et les LSTM Autoencoders.
- Optimisation des seuils de détection selon les résultats expérimentaux.
- Classification des anomalies par niveau de sévérité : `NORMAL`, `LOW`, `MEDIUM`, `HIGH` et `CRITICAL`.
- Identification des variables contribuant le plus à une anomalie, notamment à partir des erreurs de reconstruction.

**Monitoring intelligent et détection de dérive**
- Surveillance de l’évolution des données avec Evidently.
- Suivi de la qualité des prédictions et de la dégradation éventuelle des performances.
- Calcul d’un indicateur de santé du modèle (*Model Health Score*).
- Analyse des dérives afin d’identifier les situations nécessitant une investigation.

**Automatisation du cycle de vie des modèles**
- Mise en place d’un moteur de décision (*Decision Engine*) pour déterminer si un réentraînement est justifié.
- Réentraînement contrôlé des modèles lorsque les conditions définies sont réunies.
- Validation des modèles candidats avant leur promotion.
- Déploiement progressif (*Canary Deployment*) et retour à une version précédente (*Rollback*) en cas de problème.

**Industrialisation et gouvernance**
- Intégration de DVC pour le versionnement des données et la reproductibilité des expériences.
- Mise en place de pipelines CI/CD pour automatiser les étapes de validation et de déploiement.
- Étude d’une orchestration avec Kubernetes selon les besoins de déploiement.
- Amélioration de la traçabilité des expériences, des modèles et des décisions de réentraînement.

### 5.3 Principe de fonctionnement

IndustrialGuard adopte une approche de maintenance contrôlée des modèles : **la détection d’une dérive ne déclenche pas automatiquement un réentraînement**. Elle entraîne d’abord une analyse de la situation. Un nouveau modèle ne peut être déployé qu’après avoir satisfait aux critères de validation définis.
## 6. Architecture du projet

L’architecture d’IndustrialGuard repose sur une chaîne de traitement modulaire couvrant la préparation des données, l’entraînement des modèles, leur déploiement et leur supervision.

### 6.1 Vue d’ensemble

Le pipeline cible du projet est organisé selon les étapes suivantes :

1. **Données industrielles :** collecte des données du dataset HAI 23.05.
2. **Prétraitement :** nettoyage, validation et préparation des séries temporelles.
3. **Feature Engineering :** extraction de caractéristiques temporelles adaptées à la détection d’anomalies.
4. **Entraînement et évaluation :** construction et comparaison des modèles de Machine Learning.
5. **Suivi des expérimentations :** enregistrement des paramètres, métriques et versions avec MLflow.
6. **Gestion des modèles :** conservation et gestion des versions candidates dans le Model Registry.
7. **Déploiement :** exposition des prédictions à travers une API FastAPI conteneurisée avec Docker.
8. **Monitoring :** surveillance des métriques techniques et visualisation avec Prometheus et Grafana.

### 6.2 Schéma du pipeline

```text
Dataset HAI 23.05
       │
       ▼
Prétraitement des données
       │
       ▼
Feature Engineering
       │
       ▼
Entraînement et évaluation
       │
       ▼
Suivi des expériences — MLflow
       │
       ▼
Model Registry
       │
       ▼
API de prédiction — FastAPI
       │
       ▼
Conteneurisation — Docker
       │
       ▼
Monitoring — Prometheus / Grafana
```

### 6.3 Architecture évolutive

Les évolutions futures pourront compléter cette chaîne par des composants dédiés au versionnement des données avec DVC, à la détection de dérive, à l’explicabilité des anomalies et à la décision de réentraînement.

Ces composants permettront de faire évoluer le pipeline initial vers un cycle de vie MLOps plus automatisé, intégrant la validation des nouveaux modèles, leur déploiement contrôlé et le retour à une version précédente en cas de problème.

**Important :** ce schéma représente l’architecture cible du MVP. Il ne signifie pas que tous les composants sont déjà implémentés ou intégrés.
## 7. Technologies utilisées

IndustrialGuard s’appuie sur un ensemble de technologies dédiées au traitement des données, à la détection d’anomalies, au suivi des modèles et au déploiement d’une solution MLOps.

Les technologies sont organisées selon leur rôle dans le projet et leur priorité d’intégration.

### 7.1 Langages et traitement des données

| Technologie | Rôle dans IndustrialGuard | Périmètre |
|---|---|---|
| Python | Développement du pipeline de données et des modèles de détection | MVP |
| Pandas | Manipulation et préparation des données tabulaires | MVP |
| NumPy | Calcul numérique et opérations sur les séries temporelles | MVP |

### 7.2 Machine Learning et suivi des modèles

| Technologie | Rôle dans IndustrialGuard | Périmètre |
|---|---|---|
| Scikit-learn | Prétraitement, évaluation et modèle de référence Isolation Forest | MVP |
| TensorFlow / Keras ou PyTorch | Développement d’Autoencoders et de LSTM Autoencoders | Évolution |
| MLflow | Suivi des expérimentations, des métriques et des versions des modèles | MVP |
| MLflow Model Registry | Gestion des versions et du cycle de validation des modèles | MVP |
| DVC | Versionnement des données et reproductibilité des pipelines | Évolution |

### 7.3 API et déploiement

| Technologie | Rôle dans IndustrialGuard | Périmètre |
|---|---|---|
| FastAPI | Exposition des fonctionnalités de prédiction via une API REST | MVP |
| Docker | Conteneurisation de l’application et de son environnement | MVP |
| Kubernetes | Orchestration des conteneurs et gestion du déploiement à l’échelle | Évolution |
| CI/CD | Automatisation des étapes de test, de validation et de déploiement | Évolution |

### 7.4 Monitoring et détection de dérive

| Technologie | Rôle dans IndustrialGuard | Périmètre |
|---|---|---|
| Prometheus | Collecte des métriques techniques de l’application | MVP |
| Grafana | Visualisation des métriques et création de tableaux de bord | MVP |
| Evidently | Analyse de la dérive des données et suivi de la qualité des modèles | Évolution |

### 7.5 Dataset

| Ressource | Utilisation |
|---|---|
| HAI 23.05 | Jeu de données industriel utilisé pour expérimenter et évaluer la détection d’anomalies |

Le dataset est disponible sur le [dépôt officiel GitHub de HAI](https://github.com/icsdataset/hai) et sur [Kaggle](https://www.kaggle.com/datasets/icsdataset/hai-security-dataset).

## 8. Approches de Machine Learning

IndustrialGuard adoptera une approche progressive pour la détection des anomalies industrielles. L'objectif est de commencer par un modèle de référence, puis d'évaluer des approches plus avancées afin de comparer leurs performances et leur pertinence pour les données temporelles industrielles.

### 8.1. Modèle de référence : Isolation Forest

Isolation Forest sera utilisé comme première approche de détection d'anomalies. Ce modèle non supervisé permet d'identifier des observations atypiques sans nécessiter systématiquement des labels pour l'entraînement.

Il constituera une baseline permettant d'évaluer les approches plus complexes.

**Avantages attendus :**
- Mise en place relativement simple.
- Adaptation à la détection d'observations atypiques.
- Coût d'entraînement généralement modéré.
- Possibilité de comparer les résultats avec d'autres modèles.

### 8.2. Approches avancées envisagées

Deux approches complémentaires pourront être étudiées dans les évolutions du projet.

| Modèle | Principe | Objectif |
|---|---|---|
| Autoencoder | Apprendre à reconstruire les données normales | Identifier les anomalies à partir de l'erreur de reconstruction |
| LSTM Autoencoder | Apprendre les dépendances temporelles des séries de données | Détecter les comportements inhabituels dans les séquences temporelles |

Ces approches seront évaluées en fonction des ressources disponibles, de la complexité de mise en œuvre et des résultats obtenus.

### 8.3. Stratégie d'évaluation

Les performances des modèles seront évaluées à l'aide de métriques adaptées à la détection d'anomalies :

- **Precision** : proportion des alertes positives correspondant à des anomalies réelles.
- **Recall** : proportion des anomalies réelles détectées.
- **F1-score** : équilibre entre précision et rappel.
- **False Positive Rate** : proportion de données normales incorrectement signalées comme anormales.
- **Temps d'inférence** : temps nécessaire pour produire une prédiction.

Lorsque les labels de référence sont disponibles, ces métriques permettront de comparer les modèles. Une attention particulière sera accordée au déséquilibre entre les observations normales et les anomalies.

La séparation des données respectera l'ordre temporel afin de limiter les risques de fuite de données entre les ensembles d'entraînement, de validation et de test.

Aucun résultat de performance ne sera annoncé avant l'entraînement et l'évaluation effective des modèles.

---

## 9. Cycle de vie MLOps cible

IndustrialGuard vise à intégrer la détection d'anomalies dans un cycle de vie MLOps permettant de gérer les modèles de manière reproductible, traçable et contrôlée.

### 9.1. Pipeline cible

Le fonctionnement envisagé est organisé autour des étapes suivantes :

1. **Collecte des données** : chargement des fichiers du dataset HAI.
2. **Prétraitement** : vérification des données, traitement des valeurs manquantes et préparation des variables.
3. **Feature Engineering** : création et sélection des caractéristiques pertinentes pour la détection.
4. **Entraînement** : apprentissage du modèle de référence et, selon la roadmap, de modèles avancés.
5. **Évaluation** : comparaison des performances sur des données non utilisées pour l'entraînement.
6. **Tracking des expériences** : enregistrement des paramètres, métriques et artefacts avec MLflow.
7. **Gestion des versions** : conservation et identification des versions des modèles candidats.
8. **Déploiement** : exposition du modèle retenu via une API FastAPI.
9. **Monitoring** : suivi des prédictions et des indicateurs techniques disponibles.
10. **Prise de décision** : analyse des alertes et des éventuels signes de dégradation.
11. **Réentraînement contrôlé** : préparation d'un nouveau modèle lorsque des conditions prédéfinies le justifient.
12. **Validation et promotion** : vérification du nouveau modèle avant son éventuel remplacement en production.

### 9.2. Gestion du cycle de vie des modèles

Le cycle de vie cible devra distinguer plusieurs états :

- **Entraînement** : le modèle est en cours de construction.
- **Évaluation** : ses performances sont en cours de vérification.
- **Candidat** : le modèle satisfait les critères initiaux et peut être examiné pour un déploiement.
- **Déployé** : le modèle a été sélectionné et mis à disposition.
- **Retiré** : le modèle n'est plus utilisé pour les prédictions courantes, mais peut être conservé pour la traçabilité.

Les transitions entre ces états devront être définies par des règles explicites.

### 9.3. Principe de réentraînement contrôlé

Une dérive des données ou une dégradation des performances ne devra pas entraîner automatiquement le remplacement du modèle.

Le processus envisagé sera le suivant :

- Détecter un signal potentiel de dérive ou de dégradation.
- Analyser sa gravité et sa persistance.
- Déterminer si un réentraînement est justifié.
- Entraîner et évaluer un modèle candidat.
- Comparer le candidat au modèle actuellement utilisé.
- Autoriser sa promotion uniquement s'il satisfait les critères de validation.

Cette approche vise à limiter les remplacements inutiles et à réduire les risques associés au déploiement de modèles insuffisamment évalués.

---

## 10. Structure cible du dépôt

L'organisation suivante est proposée pour séparer les données, les expérimentations, le code applicatif, les tests et les éléments de déploiement.

```text
IndustrialGuard/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   └── industrialguard/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       └── monitoring/
├── api/
│   ├── main.py
│   └── schemas.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── monitoring/
│   ├── prometheus/
│   └── grafana/
├── configs/
├── docker/
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

Cette arborescence représente une **organisation cible**. Elle sera adaptée au fur et à mesure du développement et ne signifie pas que tous ces fichiers et composants existent déjà.

Les fichiers de données volumineux, les modèles entraînés et les artefacts générés ne devront pas être ajoutés directement au dépôt Git sans stratégie de stockage adaptée.

---

## 11. Installation et lancement

Les instructions d'installation définitives seront ajoutées lorsque l'environnement et les composants du MVP auront été préparés.

### 11.1. Prérequis envisagés

- Python 3.10 ou version compatible avec les dépendances retenues.
- Git.
- Un environnement virtuel Python.
- Les dépendances de traitement des données et de Machine Learning.
- Docker pour la phase de conteneurisation.
- MLflow et les outils de monitoring selon le périmètre effectivement intégré.

### 11.2. Préparation de l'environnement

Les commandes de base envisagées sont :

```bash
git clone <URL_DU_DEPOT>
cd IndustrialGuard

python -m venv .venv
```

Activation sous Windows :

```bash
.venv\Scripts\activate
```

Activation sous Linux ou macOS :

```bash
source .venv/bin/activate
```

L'installation des dépendances sera définie après la création du fichier `requirements.txt`.

**État actuel :** les commandes ci-dessus constituent une préparation indicative. Le lancement complet de l'application dépendra de l'implémentation effective du pipeline, de l'API et des services associés.

---

## 12. Roadmap du projet

Le développement sera organisé en phases progressives afin de construire d'abord une solution minimale fonctionnelle, puis d'ajouter les mécanismes MLOps avancés.

### Phase 1 — Étude et préparation des données

- [ ] Récupérer et documenter la version retenue du dataset HAI.
- [ ] Examiner la structure des fichiers et les variables disponibles.
- [ ] Identifier les labels et les périodes temporelles.
- [ ] Analyser les valeurs manquantes, les distributions et les déséquilibres.
- [ ] Définir une stratégie de séparation temporelle des données.
- [ ] Préparer le pipeline de prétraitement.

**Livrable prévu :** données documentées et pipeline de préparation reproductible.

### Phase 2 — Détection d'anomalies

- [ ] Définir les caractéristiques utilisées par les modèles.
- [ ] Entraîner un premier modèle Isolation Forest.
- [ ] Définir les seuils de décision initiaux.
- [ ] Évaluer le modèle avec les métriques retenues.
- [ ] Analyser les faux positifs et les faux négatifs.
- [ ] Documenter les limites de la baseline.

**Livrable prévu :** premier modèle de détection accompagné d'une évaluation documentée.

### Phase 3 — Tracking et gestion des modèles

- [ ] Intégrer MLflow pour suivre les expériences.
- [ ] Enregistrer les paramètres, métriques et artefacts.
- [ ] Organiser les versions des modèles candidats.
- [ ] Définir les critères de validation et de sélection.
- [ ] Étudier l'utilisation du Model Registry.

**Livrable prévu :** suivi reproductible des expériences et processus de sélection des modèles.

### Phase 4 — API et conteneurisation

- [ ] Développer une API FastAPI pour les prédictions.
- [ ] Définir les formats des requêtes et des réponses.
- [ ] Ajouter la validation des entrées et la gestion des erreurs.
- [ ] Écrire les tests de l'API.
- [ ] Conteneuriser l'application avec Docker.

**Livrable prévu :** service de prédiction accessible par API et exécutable dans un conteneur.

### Phase 5 — Monitoring initial

- [ ] Exposer les métriques techniques pertinentes.
- [ ] Configurer Prometheus pour collecter les métriques disponibles.
- [ ] Préparer des tableaux de bord Grafana.
- [ ] Suivre notamment la latence, le volume des requêtes et les erreurs.
- [ ] Étudier le suivi des prédictions et des alertes.

**Livrable prévu :** première version du monitoring technique et applicatif.

### Phase 6 — Évolutions avancées

- [ ] Comparer Isolation Forest avec un Autoencoder.
- [ ] Étudier un LSTM Autoencoder pour les séries temporelles.
- [ ] Ajouter la détection de dérive avec Evidently ou une solution équivalente.
- [ ] Concevoir un moteur de décision pour les alertes et les demandes de réentraînement.
- [ ] Définir un processus de validation et de promotion des nouveaux modèles.
- [ ] Étudier le déploiement progressif, le rollback et le versionnement des données avec DVC.
- [ ] Évaluer l'intégration d'une chaîne CI/CD et de Kubernetes.

**Livrable prévu :** architecture MLOps enrichie, selon le temps disponible et les résultats des phases précédentes.

Les cases à cocher de cette roadmap représentent des tâches prévues. Elles seront mises à jour au fur et à mesure de leur réalisation.

---

## 13. Utilisation responsable et limites

IndustrialGuard est un projet académique destiné à étudier la détection d'anomalies et la gestion du cycle de vie des modèles dans un contexte industriel.

Plusieurs limites devront être prises en compte :

- **Représentativité des données :** le dataset HAI est un environnement expérimental et ne couvre pas nécessairement toutes les conditions d'un système industriel réel.
- **Faux positifs et faux négatifs :** une alerte peut signaler à tort un fonctionnement normal, ou ne pas détecter une anomalie réelle.
- **Dérive des données :** les comportements appris peuvent devenir moins pertinents lorsque les conditions opérationnelles évoluent.
- **Qualité de l'évaluation :** les résultats dépendent notamment de la séparation temporelle, du choix des variables et de la qualité des labels.
- **Ressources de calcul :** les modèles profonds et les mécanismes MLOps avancés peuvent nécessiter davantage de ressources.
- **Sécurité opérationnelle :** un modèle de détection ne doit pas être considéré comme un mécanisme de sécurité suffisant à lui seul.

Dans un contexte industriel réel, les alertes et les propositions de réentraînement devraient être soumises à des règles de validation adaptées. Toute décision opérationnelle critique nécessiterait des vérifications supplémentaires et une supervision humaine appropriée.

---

## Conclusion

IndustrialGuard propose une démarche progressive associant détection d'anomalies, suivi des modèles, API de prédiction et monitoring. Le MVP permettra de construire une première chaîne fonctionnelle, tandis que les évolutions envisagées viseront à renforcer la détection de dérive, la validation des modèles et la maîtrise de leur cycle de vie.

L'objectif final est de concevoir une architecture MLOps reproductible, évolutive et adaptée aux contraintes d'un cas d'étude industriel, sans supposer que les résultats ou les composants futurs sont déjà opérationnels.
