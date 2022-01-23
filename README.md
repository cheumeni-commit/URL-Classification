# URL-Classification

# Description

Le but de ce projet est de construire un modèle de ML, qui permet de classifier des URL.

Nous disposons de 5 datatset, qui fusionnés, permettent d'obtenir un unique dataset ayant les caracteristiques suivantes:
- 67595 lignes et 3 colonnes dont les entêtes sont : 'url', 'target' et 'day'
- il possède 31253 labels contenant une ou plusieurs classe parmi les 1676 classes indexées
- les labels, éléments de la colonne 'target', sont très déséquilibrées. En effet, nous avons 90% des url qui ont moins de 150 occurences
- l'analyse de la distribution des occurences des labels montre une asymétrie vers la droite (un fort skewness)
- avec une taille de vecteur encoder (embbeding) de 500 et un seuil d'occurence à 150 (occurrences min des labels), nous obtenons les resultats suivant:
    "precision": 0.81, "recall": 0.82, "f1": 0.80,
- nous n'avons pas utilisé de technique de 'over-sample' comme par exemple 'SMOTE' pour augmenter les données des classes fiables. En effet, faire cela revient à créer des 'url' artificielles pour l'entraînement du modèle. Or ces données artificielles, risques de ne pas être réprésentatives des 'url' réelles associées à ces différentes classes et par conséquent, peuvent introduire un biais lors de l'apprentissage (pour plus d'info, lire la conclusion de ce travail : https://www.kaggle.com/theoviel/dealing-with-class-imbalance-with-smote) et ainsi dégrader les perfs lors de la prédiction.

# Modelisation

Le problème à modéliser appartient à la famille des problèmes de Classification supervisée rencontrés en NLP (Natural Language Processing)
Pour ce faire :

- nous avons transformé toutes les lignes de texte 'url' en texte classique en supprimant certains mots indiqués dans le fichier constant.py
- une fois le texte transformé, nous l'avons décomposé en mot et l'avons encodé en vecteur binaire grâce à la fonction Count_Vectorizer()
- nous avons sauvé le lexique de mots obtenu pour l'encodage lors de la phase d'apprentissage, afin de l'utiliser pour la prédiction
- le déséquilibre de classe a été géré avec le 'stratify' de la fonction train_test_split()
- nous avons utilisé trois types de classifier ML (de sklearn et xgboost) : RandomForestClassifier, xgb_Classifier et SGDClassifier
- nous avons deux modes :
    - dev : utilisation des classifiers sans optimisation des parametres (non utilisation de GridSearchCV)
    - prod : avec utilisation du GridSearchCV ou RandomSearchCV et le choix du meilleur classifier + sauvegarde des poids (serialisation)
- comme metrique d'évaluation, nous avons utilisé:
    - dev : les métriques de 'précision', 'recall' et 'f1_score' pour l'évaluation sur la base de test
    - prod : la métrique 'roc_auc' pour l'optimisation et les métriques de 'précision', 'recall' et 'f1_score' pour l'évaluation sur la base de test et le choix du meilleur modèle

# Installations

Pour faire fonctionner le projet, il faut soit réaliser un 'fork' ou le 'cloner', ensuite :
- installer les libs contenues dans le fichier 'requirements.txt' dans un 'venv', et taper en ligne de commande : 
```
pip install -e .
```

# Requirements

- lire le fichier didié

## Utilisation

- fusionner et générer le dataset intermédaire pour l'apprentissage :
```
python3 -m src run_dataset --e dev (ou 'prod')
```
- réaliser l'apprentissage des différents modèles de classification : 
```
python3 -m src run_train --e dev (ou 'prod)
```
- réaliser l'inférence ou la prédiction du meilleur modèle sauvé : 
```
python3 -m src run_predict --e dev (ou 'prod')
```

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