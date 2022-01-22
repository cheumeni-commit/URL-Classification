# URL-Classification

# Description

Le but de ce projet est de construire un modèle de ML, qui permet de classifier des URL.

Nous disposons de plusieurs mini datatset, qui fusionnés, permet d'obtenir un dataset ayant les caracteristiques suivantes:
- un dataset de 67595 lignes et de 3 colonnes dont les entêtes sont : 'url', 'target' et 'day'
- le dataset possède 31253 labels contenant une ou plusieurs classe parmi les 1676 classes indexées
- les classes, éléments de la colonne 'target', sont très déséquilibrées. En effet, nous avons 90% des url qui ont moins de 150 occurences
- l'analyse de la distribution des occurences des classe montre une asymétrie vers la droite (fort skewness)
- avec une taille de vecteur encoder (embbeding) de 500 et un seuil de 150 (occurrences min des labels), nous avons le resultats suivant:
    "precision": 0.81, "recall": 0.82, "f1": 0.80,

# Modelisation

Le problème à modéliser appartient à la famille de problème de Classification rencontrés en NLP (Natural Language Processing)
Pour ce faire :

- nous avons transformé toutes les ligne de texte 'url' en texte classique en supprimant certains mots indiqués dans le fichier constant.py
- une fois le texte transformé, nous l'avons décomposé en mot et l'avons encodé en vecteur binaire grâce à la fonction Count_Vectorizer()
- nous avons sauvé le lexique de mots obtenu pour l'encodage lors de la phase de prédiction, une fois le réseau appris et figé
- nous avons utilisé trois type de classifier ML (de sklearn et xgboost) : RandomForestClassifier, xgb_Classifier et SGDClassifier
- nous avons deux modes :
    - dev : utilisation des classifier sans optimisation des parametres (non utilisation de GridSearchCV)
    - prod : avec utilisation de GridSearchCV ou RandomSearchCV et choix du meilleur classifier + sauvegarde des poids (serialisation)
- comme metrique d'évaluation, nous avons utilisé:
    - dev : les métriques 'précision', 'recall' et 'f1_score' pour l'évalution sur la base de test
    - prod : la métrique 'roc_auc' pour l'optimisation et les métriques 'précision', 'recall' et 'f1_score' pour l'évalution sur la base de test et choix du meilleur modèle

# Installations

Pour faire fonctionner le projet, il faut soit réaliser un 'fork' ou le 'cloner'
- pour installer les libs dans le fichier 'requirements.txt' dans le venv, il faut taper en ligne de commande : pip install -e . 

# Requirements

- lire le fichier didié

## Utilisation

- fusionner et générer le dataset intermédaire pour l'apprentissage : python3 -m src run_dataset --e dev (ou 'prod')
- réaliser l'apprentissage des différents modèles de classification : python3 -m src run_train --e dev (ou 'prod)
- réaliser l'inférence ou la prédiction du meilleur modèle sauvé : python3 -m src run_predict --e dev (ou 'prod')

# ChangeLog

- V.1.0.0 : Release intiale

# Arborescence du projet

- classique :
    - config
    - data
    - src

# Citing ML URL-Classification

Si vous utilisez ce projet pour faire des travaux de recherche et pour résoudre d'autres problèmes, merci de me citer comme suit.

```BibTeX
@misc{jmc2021GenerativeIA,
  author =       {Jean-Michel Cheumeni},
  title =        {Generative AI},
  howpublished = {\url{git@github.com:cheumeni-commit/URL-Classification.git}},
  year =         {2021}
}
```