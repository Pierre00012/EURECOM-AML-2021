# Cahier des charges
## Projet : Glaucoma Detection — Pipeline Machine Learning & MLOps de bout en bout

**Version :** 1.0  
**Type de projet :** Computer Vision / Machine Learning / MLOps  
**Domaine :** Santé — analyse d'images rétiniennes  
**Environnement principal :** Google Colab + Google Drive  
**Déploiement cible :** API REST avec FastAPI, conteneur Docker et Google Cloud Run en option  
**Dataset :** REFUGE — compétition Kaggle EURECOM AML 2021 Challenge 2

---

# 1. Contexte et problématique

Le glaucome est une maladie oculaire pouvant entraîner une perte irréversible de la vision. Son identification précoce est importante, mais son évaluation clinique nécessite plusieurs éléments, et pas uniquement une photographie du fond d'œil.

Le projet consiste à développer un prototype de machine learning capable d'analyser une photographie rétinienne et d'estimer la probabilité qu'elle soit associée à un cas de glaucome.

Le projet doit couvrir l'intégralité du cycle de vie d'un modèle ML :

- Recherche bibliographique et compréhension métier.
- Acquisition et gestion des données.
- Analyse exploratoire et contrôle qualité.
- Prétraitement et préparation des données.
- Entraînement et comparaison de modèles.
- Évaluation et interprétation des résultats.
- Suivi et traçabilité des expériences.
- Gestion des versions des données et du modèle.
- Automatisation des tests.
- Packaging et déploiement.
- Exposition du modèle via une API REST.
- Monitoring et documentation.

**Positionnement :** prototype expérimental d'aide à l'analyse d'images, et non dispositif médical, diagnostic ou outil de décision clinique.

# 2. Objectifs du projet

## 2.1 Objectif principal

Développer un pipeline reproductible permettant de passer d'un jeu de données brut à un modèle entraîné, évalué, versionné et accessible via une API REST.

## 2.2 Objectifs techniques

1. Organiser un projet ML selon une architecture logicielle claire.
2. Exécuter les entraînements sur Google Colab.
3. Conserver les données, modèles et résultats dans Google Drive.
4. Comparer plusieurs approches de classification.
5. Visualiser les métriques d'entraînement et de validation.
6. Enregistrer chaque expérience et ses hyperparamètres.
7. Automatiser les tests du code et du modèle.
8. Construire une API REST pour effectuer des prédictions.
9. Conteneuriser l'application avec Docker.
10. Déployer l'API et documenter son utilisation.

## 2.3 Objectifs pédagogiques

Le projet doit permettre de comprendre et de pratiquer :

- Les fondamentaux du Deep Learning appliqué aux images.
- Le transfert d'apprentissage.
- Les problématiques de déséquilibre de classes.
- La reproductibilité expérimentale.
- Le versionnement des données et des modèles.
- Le tracking des expériences.
- Le CI/CD appliqué au ML.
- Le monitoring d'une application ML.
- Les limites de généralisation d'un modèle médical.

# 3. Périmètre fonctionnel

## 3.1 Version initiale — classification

L'application prend en entrée une photographie du fond d'œil et retourne :

- Une probabilité estimée de glaucome.
- Une classe prédite selon un seuil configurable.
- La version du modèle utilisée.
- Le temps d'inférence.
- Un statut indiquant si l'image a pu être traitée.

La sortie doit clairement indiquer qu'il s'agit d'une estimation expérimentale et non d'un diagnostic médical.

## 3.2 Extensions facultatives

Après validation de la classification :

- Segmentation du disque optique.
- Segmentation de la cup optique.
- Calcul du Cup-to-Disc Ratio.
- Visualisation Grad-CAM.
- Comparaison de modèles multitâches.

Ces extensions ne doivent pas retarder la réalisation du pipeline MLOps principal.

# 4. Dataset et recherche bibliographique

## 4.1 Source principale

Compétition Kaggle :

https://www.kaggle.com/competitions/eurecom-aml-2021-challenge-2

Documentation officielle REFUGE :

https://refuge.grand-challenge.org/Details/

Article scientifique de référence :

https://pubmed.ncbi.nlm.nih.gov/31630011/

Le dataset REFUGE contient 1 200 photographies rétiniennes réparties en trois ensembles de 400 images. Les cas positifs représentent environ 10 % des exemples.

Les annotations peuvent comprendre :
- Le statut glaucomateux.
- Les masques du disque optique et de la cup.
- Les coordonnées de la fovéa.

**Point de contrôle obligatoire :** vérifier la structure réelle des fichiers, les labels disponibles et les conditions d'utilisation du dataset. Les annotations du test ne doivent pas être supposées accessibles.

## 4.2 Travail bibliographique

Avant de développer les modèles, réaliser une revue de littérature comprenant au minimum :

1. Compréhension du glaucome et des photographies du fond d'œil.
2. Méthodes classiques de classification d'images médicales.
3. CNN et transfert d'apprentissage.
4. ResNet et EfficientNet.
5. Méthodes de gestion du déséquilibre de classes.
6. Métriques adaptées à la classification médicale.
7. Explainable AI et Grad-CAM.
8. Domain shift et généralisation inter-datasets.
9. Bonnes pratiques MLOps pour le Deep Learning.

Pour chaque article lu, conserver une fiche :
- Référence bibliographique.
- Problématique.
- Dataset utilisé.
- Méthodologie.
- Architecture.
- Métriques.
- Résultats.
- Limites.
- Idées à reproduire ou à tester.

# 5. Architecture technique

## 5.1 Environnement de développement

**Entraînement :**
- Google Colab.
- GPU Colab lorsque disponible.
- Google Drive pour la persistance des fichiers.

**Langage et bibliothèques :**
- Python.
- Pandas et NumPy.
- Matplotlib et Seaborn.
- OpenCV et Pillow.
- Scikit-learn.
- PyTorch et Torchvision.

**MLOps :**
- MLflow : tracking des expériences et gestion des modèles.
- TensorBoard : visualisation des courbes d'entraînement.
- DVC : versionnement des données et artefacts, si pertinent.
- Git et GitHub : versionnement du code.
- Pytest : tests automatisés.
- GitHub Actions : intégration continue.

**API et déploiement :**
- FastAPI.
- Pydantic.
- Uvicorn.
- Docker.
- Google Cloud Run en option.
- Swagger UI pour tester les endpoints.

## 5.2 Architecture logique

Données brutes  
→ Validation et contrôle qualité  
→ Prétraitement  
→ Dataset versionné  
→ Entraînement  
→ Tracking des expériences  
→ Évaluation  
→ Sélection du modèle  
→ Enregistrement du modèle  
→ Tests  
→ Packaging Docker  
→ API REST  
→ Déploiement  
→ Monitoring

# 6. Pipeline détaillée

## Phase 1 — Initialisation et gestion du projet

### Travaux à réaliser

- Créer un dépôt GitHub.
- Définir la structure du projet.
- Configurer l'environnement Python.
- Connecter Google Colab à Google Drive.
- Définir les dépendances dans un fichier requirements.txt.
- Créer un fichier de configuration centralisé.
- Définir une convention de nommage des expériences et des modèles.

### Livrables

- Dépôt GitHub.
- README initial.
- requirements.txt.
- Configuration du projet.
- Notebook de vérification de l'environnement.

### Critères d'acceptation

- Le notebook peut être exécuté dans un environnement Colab propre.
- Les fichiers persistants sont enregistrés dans Google Drive.
- Le projet est organisé et documenté.

## Phase 2 — Acquisition et gestion des données

### Travaux à réaliser

- Télécharger les données depuis la source autorisée.
- Décompresser et organiser les fichiers.
- Vérifier les images et annotations disponibles.
- Construire un inventaire des fichiers.
- Vérifier les doublons.
- Vérifier les images corrompues.
- Vérifier la correspondance entre images et labels.
- Identifier les dimensions et formats.
- Vérifier la répartition des classes.
- Conserver une trace de la version des données.

### Contrôles obligatoires

- Nombre d'images.
- Nombre d'images positives et négatives.
- Images manquantes.
- Labels manquants.
- Images illisibles.
- Dimensions et formats.
- Doublons exacts ou quasi-identiques.
- Cohérence entre les splits.

### Livrables

- Script d'acquisition.
- Rapport de qualité des données.
- Inventaire CSV.
- Version de référence du dataset.
- Documentation de provenance.

### Critères d'acceptation

- Toutes les images utilisées ont un label vérifié.
- Aucun fichier corrompu n'est utilisé silencieusement.
- Les données de test ne sont pas utilisées pour entraîner ou sélectionner le modèle.

## Phase 3 — Exploratory Data Analysis

### Travaux à réaliser

Étudier :
- La distribution des classes.
- Les dimensions des images.
- Les différences de résolution.
- Les variations de luminosité et de contraste.
- Les variations de couleur.
- Les artefacts et images de mauvaise qualité.
- Les éventuelles différences entre appareils d'acquisition.

### Visualisations obligatoires

- Répartition des classes.
- Exemples d'images par classe.
- Histogrammes des dimensions.
- Distribution des canaux RGB.
- Distribution de luminosité.
- Comparaison visuelle des images.
- Matrice de corrélation des variables numériques pertinentes.

### Livrables

- Notebook EDA.
- Rapport illustré.
- Conclusions sur les risques liés aux données.

### Critères d'acceptation

Les choix de prétraitement doivent être justifiés par les observations de l'EDA.

## Phase 4 — Préparation et prétraitement

### Travaux à réaliser

- Redimensionner les images.
- Convertir les images dans le format attendu.
- Normaliser les valeurs des pixels.
- Définir les transformations d'entraînement.
- Définir les transformations de validation et de test.
- Construire les DataLoaders PyTorch.

### Augmentation de données

Étudier et tester :
- Rotations modérées.
- Flips horizontaux si justifiés.
- Modifications de luminosité.
- Modifications de contraste.
- Transformations géométriques raisonnables.

Ne jamais appliquer d'augmentation aléatoire aux données de validation ou de test.

### Gestion du déséquilibre

Comparer :
- Pondération des classes.
- Weighted loss.
- Éventuellement focal loss.

Ne pas considérer l'accuracy comme métrique principale.

### Livrables

- Module preprocessing.
- Module dataset.
- Configuration des transformations.
- Tests de préparation des données.

## Phase 5 — Modélisation et entraînement

### Modèles à comparer

**Modèle 1 — Baseline :**
- CNN simple construit avec PyTorch.

**Modèle 2 — Transfert d'apprentissage :**
- ResNet18 pré-entraîné.

**Modèle 3 — Modèle avancé :**
- EfficientNet-B0 ou architecture équivalente.

Le but est de comparer les modèles selon un protocole identique.

### Protocole expérimental

Pour chaque entraînement, enregistrer :
- Nom du modèle.
- Version du code.
- Version des données.
- Hyperparamètres.
- Optimiseur.
- Learning rate.
- Batch size.
- Nombre d'epochs.
- Fonction de perte.
- Seed.
- Durée d'entraînement.
- Métriques train et validation.
- Chemin du checkpoint.

### Visualisation des métriques

Utiliser TensorBoard et/ou MLflow pour suivre :

- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.
- ROC-AUC.
- PR-AUC.
- Precision.
- Recall.
- F1-score.
- Sensibilité.
- Spécificité.
- Matrice de confusion.
- Learning rate.
- Durée des expériences.

Les courbes de loss et de métriques doivent permettre d'identifier le surapprentissage.

### Gestion des expériences

MLflow doit permettre :
- De créer une expérience par famille de modèles.
- De comparer plusieurs runs.
- De retrouver les hyperparamètres.
- De consulter les métriques.
- De stocker les artefacts.
- D'identifier le meilleur modèle selon un critère défini.

Attention : les sessions Colab sont temporaires. Les fichiers de tracking et les artefacts doivent être sauvegardés dans un emplacement persistant. Ne pas dépendre uniquement du stockage local de la session.

### Livrables

- Scripts d'entraînement.
- Configuration des expériences.
- Courbes TensorBoard.
- Tableau comparatif MLflow.
- Checkpoints des modèles.
- Rapport d'entraînement.

## Phase 6 — Évaluation et sélection du modèle

### Métriques principales

**ROC-AUC :** mesure la capacité du modèle à distinguer les deux classes sur différents seuils.

**PR-AUC :** particulièrement importante compte tenu du déséquilibre des classes.

**Sensibilité (Recall positif) :** proportion des cas positifs correctement détectés.

**Spécificité :** proportion des cas négatifs correctement identifiés.

**Precision :** proportion des prédictions positives qui sont réellement positives.

**F1-score :** compromis entre précision et rappel.

**Matrice de confusion :** permet d'étudier les faux positifs et faux négatifs.

### Analyses complémentaires

- Courbe ROC.
- Courbe Precision-Recall.
- Matrice de confusion.
- Comparaison des seuils.
- Analyse des erreurs.
- Exemples de faux positifs.
- Exemples de faux négatifs.
- Grad-CAM pour visualiser les régions influençant certaines prédictions.

### Règles d'évaluation

- Définir le critère principal avant la comparaison finale.
- Choisir le seuil à partir de la validation.
- Ne pas choisir le modèle en fonction des résultats du test.
- Rapporter les métriques avec prudence et, si possible, des intervalles de confiance.
- Documenter les limites dues à la taille du dataset.

### Livrables

- Rapport d'évaluation.
- Tableau comparatif des modèles.
- Graphiques des métriques.
- Analyse des erreurs.
- Modèle candidat sélectionné.

## Phase 7 — Reproductibilité et versionnement

### Objectif

Pouvoir reproduire un entraînement et retrouver exactement les éléments qui ont produit un modèle.

### Travaux à réaliser

- Versionner le code avec Git.
- Versionner les données ou leurs références avec DVC.
- Sauvegarder les configurations.
- Fixer les seeds lorsque possible.
- Enregistrer les versions des bibliothèques.
- Enregistrer les paramètres et résultats avec MLflow.
- Sauvegarder les checkpoints.
- Créer une fiche descriptive du modèle.

### Model card

La fiche modèle doit contenir :
- Nom et version.
- Dataset et provenance.
- Prétraitement.
- Architecture.
- Hyperparamètres.
- Métriques.
- Seuil utilisé.
- Limites connues.
- Usages prévus et interdits.
- Date de création.

### Critère d'acceptation

Un entraînement doit pouvoir être relancé à partir du code, de la configuration et de la version des données documentées.

## Phase 8 — Tests automatisés

### Tests unitaires

Tester :
- Le chargement d'une image.
- Le prétraitement.
- La forme des tenseurs.
- La sortie du modèle.
- Le calcul des métriques.
- La validation des entrées API.

### Tests d'intégration

Tester :
- Le chargement du modèle sauvegardé.
- Le pipeline complet d'inférence.
- La cohérence des sorties.
- La gestion des images invalides.

### Tests de non-régression

Conserver quelques exemples de référence et vérifier que les changements de code ne provoquent pas de régression inattendue.

### Tests API

Tester :
- Le code HTTP.
- Le format JSON.
- Les erreurs d'entrée.
- Le comportement lorsque le modèle est absent.
- Le temps de réponse dans un environnement donné.

### Outils

- Pytest.
- FastAPI TestClient.
- GitHub Actions.

### Critère d'acceptation

Les tests critiques doivent être exécutés automatiquement à chaque pull request ou modification importante.

## Phase 9 — API REST

### Objectif

Exposer le modèle entraîné via une petite API REST permettant de tester des images.

### Technologies

- FastAPI.
- Uvicorn.
- Pydantic.
- PyTorch.
- Pillow.

### Endpoints attendus

**GET /health**

Retourne l'état de l'application.

Exemple :
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**GET /model-info**

Retourne les informations du modèle chargé.

Exemple :
```json
{
  "model_name": "resnet18_glaucoma",
  "model_version": "1.0.0",
  "task": "binary_classification"
}
```

**POST /predict**

Accepte une image au format multipart/form-data et retourne :

```json
{
  "prediction": 1,
  "probability": 0.82,
  "threshold": 0.50,
  "model_version": "1.0.0",
  "inference_time_ms": 125.4,
  "notice": "Experimental result, not a medical diagnosis."
}
```

Les valeurs de cet exemple sont fictives et servent uniquement à illustrer le format de réponse.

### Validation des entrées

- Accepter uniquement les formats autorisés.
- Vérifier que le fichier est une image lisible.
- Limiter la taille du fichier.
- Gérer les images corrompues.
- Retourner des erreurs HTTP explicites.
- Ne pas enregistrer les images envoyées par défaut.

### Documentation

FastAPI doit exposer Swagger UI via :

`/docs`

Cette interface doit permettre d'envoyer une image et de consulter la réponse de l'API.

## Phase 10 — Dockerisation

### Travaux à réaliser

- Créer un Dockerfile.
- Installer les dépendances.
- Copier le code nécessaire.
- Charger le modèle depuis un emplacement configurable.
- Exposer le port de l'application.
- Définir la commande de démarrage.
- Vérifier le fonctionnement du conteneur.

### Critères d'acceptation

- L'API démarre dans le conteneur.
- Le modèle est chargé correctement.
- Les endpoints fonctionnent.
- Les tests d'intégration passent.

## Phase 11 — CI/CD

### Objectif

Automatiser les vérifications du code et préparer les livraisons.

### Pipeline GitHub Actions

À chaque pull request :

1. Récupération du code.
2. Installation des dépendances.
3. Vérification de la qualité du code.
4. Exécution des tests unitaires.
5. Exécution des tests d'API.
6. Construction de l'image Docker.
7. Vérification de l'image.

Lorsqu'une version est validée :

1. Création d'un tag Git.
2. Construction de l'image Docker.
3. Publication de l'image dans un registre.
4. Déploiement de l'API.
5. Vérification du endpoint health.

Le CI/CD ne doit pas entraîner automatiquement un modèle coûteux à chaque commit. L'entraînement et le déploiement doivent être deux étapes distinctes et contrôlées.

## Phase 12 — Déploiement

### Environnement de test

Exécuter localement le conteneur Docker et tester l'API avec Swagger UI ou un client HTTP.

### Environnement distant

Option recommandée : Google Cloud Run.

Étapes :
- Créer un projet Google Cloud.
- Configurer les permissions.
- Publier l'image dans Artifact Registry.
- Déployer le conteneur sur Cloud Run.
- Configurer les variables d'environnement.
- Vérifier les endpoints.
- Tester l'inférence à distance.

Le modèle doit être accessible au démarrage du service, soit intégré au conteneur si sa taille le permet, soit récupéré depuis un stockage configuré.

Les identifiants et secrets ne doivent jamais être stockés dans GitHub ou dans le code source.

### Livrables

- Dockerfile.
- Image Docker.
- Configuration de déploiement.
- URL de l'API si le déploiement distant est réalisé.
- Documentation de test.

## Phase 13 — Monitoring et maintenance

### Monitoring technique

Suivre :
- Nombre de requêtes.
- Temps de réponse.
- Taux d'erreurs.
- Disponibilité de l'API.
- Consommation des ressources.
- Temps d'inférence.

### Monitoring ML

Prévoir :
- Distribution des probabilités prédites.
- Distribution des classes prédites.
- Évolution des scores lorsque des labels de référence deviennent disponibles.
- Détection de changements dans les données d'entrée.
- Comparaison des performances sur de nouveaux jeux annotés.

Il faut distinguer le data drift du concept drift. Une variation des images ne prouve pas à elle seule une dégradation des performances.

Sans labels réels, il est impossible de mesurer directement la performance clinique en production. Le monitoring doit donc rester transparent sur ce point.

### Maintenance

- Versionner chaque nouveau modèle.
- Conserver les modèles précédents.
- Définir une procédure de rollback.
- Réentraîner uniquement après analyse et validation.
- Documenter chaque changement.

# 7. Organisation du dépôt GitHub

Structure cible :

```text
glaucoma-mlops/
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── architectures.py
│   │   ├── train.py
│   │   └── evaluate.py
│   │
│   ├── tracking/
│   │   └── mlflow_utils.py
│   │
│   └── inference/
│       └── predictor.py
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── artifacts/
│   └── README.md
│
├── reports/
│   └── README.md
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
├── .gitignore
└── dvc.yaml
```

Les données, checkpoints volumineux et secrets ne doivent pas être commités directement dans Git.

# 8. Planification indicative

Durée cible : 10 semaines, à adapter au temps disponible.

| Semaine | Travail | Livrable |
|---|---|---|
| 1 | Recherche bibliographique et compréhension métier | Fiches de lecture |
| 2 | Acquisition, contrôle qualité et EDA | Rapport EDA |
| 3 | Prétraitement et baseline | Pipeline de données |
| 4 | Entraînement ResNet18 | Première expérience MLflow |
| 5 | EfficientNet et comparaison | Tableau comparatif |
| 6 | Évaluation, seuils et Grad-CAM | Rapport d'évaluation |
| 7 | Versionnement, tests et reproductibilité | Tests et Model Card |
| 8 | API FastAPI et tests | API fonctionnelle |
| 9 | Docker, CI/CD et déploiement | Application conteneurisée |
| 10 | Monitoring, documentation et démonstration | Projet final |

# 9. Critères de réussite globaux

Le projet sera considéré comme terminé lorsque :

- [ ] Le dataset est documenté et contrôlé.
- [ ] L'EDA est complète et interprétée.
- [ ] Un modèle baseline est entraîné.
- [ ] Au moins deux architectures sont comparées.
- [ ] Les métriques sont visualisables.
- [ ] Les expériences sont tracées.
- [ ] Le meilleur modèle est sauvegardé et versionné.
- [ ] L'entraînement est reproductible autant que possible.
- [ ] Les tests unitaires et d'intégration fonctionnent.
- [ ] L'API REST fonctionne.
- [ ] L'API est documentée avec Swagger.
- [ ] Le projet est dockerisé.
- [ ] Une pipeline CI exécute les tests.
- [ ] Le déploiement distant est documenté ou réalisé.
- [ ] Le monitoring technique est défini.
- [ ] Les limites scientifiques et médicales sont explicitées.

# 10. Règles de travail et autonomie

Le projet doit être conduit comme un travail de recherche et d'ingénierie personnel.

L'utilisation de l'IA est volontairement limitée.

**L'IA peut être utilisée pour :**
- Clarifier une notion technique.
- Expliquer une erreur précise.
- Discuter une décision d'architecture.
- Aider à comprendre une documentation.
- Donner des pistes de recherche bibliographique.
- Challenger une hypothèse ou une interprétation.

**L'IA ne doit pas être utilisée pour :**
- Générer tout le projet sans compréhension.
- Remplacer la lecture des articles.
- Produire des conclusions scientifiques non vérifiées.
- Choisir un modèle sans expérimentation.
- Inventer des résultats ou des métriques.
- Rédiger artificiellement un rapport qui ne reflète pas le travail effectué.

Chaque choix important doit être justifié dans le README ou dans un document de décision technique.

# 11. Livrables finaux du portfolio

Le dépôt GitHub devra présenter :

1. Un README complet avec la problématique et l'architecture.
2. Les références scientifiques étudiées.
3. Le pipeline de données.
4. Les notebooks d'analyse.
5. Les scripts d'entraînement.
6. Les visualisations des métriques.
7. Les résultats comparatifs.
8. La Model Card.
9. Le code de l'API REST.
10. Le Dockerfile.
11. Les workflows CI/CD.
12. Les instructions de déploiement.
13. Une démonstration de l'API.
14. Une section consacrée aux limites et aux pistes d'amélioration.

**Résultat attendu :** un projet de portfolio démontrant non seulement la capacité à entraîner un modèle de Deep Learning, mais aussi à construire, tester, versionner, déployer et maintenir une application basée sur le Machine Learning.