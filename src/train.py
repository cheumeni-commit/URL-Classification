import logging
from matplotlib.pyplot import axis

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import SMOTE

from src.cli import context
from src.training.features import build_train_test_set
from src.training.evaluation import evaluate_model
from src.io import save_lexique
from src.constants import (c_SIZE,
                           c_TEXT_TRANSFORMES,
                           c_TAGS,
                           c_DETAIL_ECRITURE,
                           c_BOX_MAX_FEATURES,
                           c_FEATURES_PERCENTILES,
                           c_SEED,
                           c_MIN_LABEL_COUNT,
                           c_LEXIQUE,
                           c_DAY,
                           c_LABELS
                          )


logger = logging.getLogger(__name__)


def _split_target(features_set):
    np.random.random(200)
    # Separate dataset into training and test
    (X_train, X_test, y_train, y_test) = _split_train_test(features_set)
    # Over_Sample
    #oversample = SMOTE(sampling_strategy='minority')
    #X_train, y_train = oversample.fit_resample(X_train, y_train)
    # label encoder
    y_train, y_test = _label_encoder(y_train, y_test)
    # generation bag of word
    (X_train_bow, X_test_bow) = _bow_transformation(X_train, X_test)
    # Extract features
    (X_train_features, X_test_features) = _extract_features(X_train_bow,
                                                             X_test_bow,
                                                              y_train)

    return X_train_features, y_train, X_test_features, y_test


def _label_encoder(y_train, y_test):
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    label_encoder.fit(y_test)
    y_train = label_encoder.encode(y_train)
    y_test = label_encoder.encode(y_test)
    return y_train, y_test


def _split_train_test(features_set):
    
    X = features_set.loc[:, c_TEXT_TRANSFORMES].values
    y = features_set.loc[:, c_TAGS].values
   
    # use stratify if min individu per class is sup than 10
    #https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
    if np.min(np.unique(y, return_counts=True)[1]) >=c_MIN_LABEL_COUNT:
        return train_test_split(X, y, 
                                test_size = c_SIZE,
                                 random_state=c_SEED
                                 )
    else: 
        return train_test_split(X, y, 
                                test_size = c_SIZE, 
                                random_state=c_SEED
                                )


def _bow_transformation(X_train, X_test):
    vectorizer = Count_Vectorizer()
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    save_lexique(vectorizer.get_feature_names(),
                 path=context.dirs.config / c_LEXIQUE
                 )
    return X_train_bow.toarray(), X_test_bow.toarray()


def Count_Vectorizer():
    return CountVectorizer(max_features=c_BOX_MAX_FEATURES,
                           strip_accents='unicode'
                           )


def Select_Percentile():
    return SelectPercentile(f_classif, 
                            percentile = c_FEATURES_PERCENTILES
                            )


def _extract_features(X_train_bow, X_test_bow, y_train):
    feature_selector = Select_Percentile()
    X_train_features = feature_selector.fit_transform(X_train_bow,
                                                      y_train
                                                      )
    X_test_features = feature_selector.transform(X_test_bow)
    return X_train_bow, X_test_bow


class LabelEncoder(object):
    """Label encoder for tag labels."""
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y):
        classes = np.unique(y)
        for i, class_ in enumerate(classes):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        save_lexique(self.index_to_class, path=context.dirs.config / c_LABELS)
        self.classes = list(self.class_to_index.keys())
        return self

    def encode(self, y):
        encoded = np.zeros((len(y)), dtype=int)
        for i, item in enumerate(y):
            encoded[i] = self.class_to_index[item]
        return encoded


def algorithm_pipeline(GridMethod, X_train_data, 
                        y_train_data, 
                       model, param_grid, cv=5, 
                       scoring_fit='roc_auc'
                       ):
    
    if GridMethod == "GridSearchCV":
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            scoring=scoring_fit,
            verbose=1
        )
        fitted_model = gs.fit(X_train_data, y_train_data)
        
    else:
        cs = RandomizedSearchCV(
            estimator=model,
            param_grid=param_grid, 
            cv=cv, 
            n_jobs=-1, 
            scoring=scoring_fit,
            verbose=1  
        )
        fitted_model = cs.fit(X_train_data, y_train_data)
    
    return fitted_model


def __train_model(model, X, y, param_grid):
    return algorithm_pipeline("GridSearchCV", X, y, 
                       model, param_grid
                       )
            
def _train_model(model, X, y):
    model.fit(X, y) 
    return model


def train(models, name_models, dataset):

    # transformation of transactions lines
    dataset_with_feature = build_train_test_set(dataset, 
                                                column_ecriture=c_DETAIL_ECRITURE
                                                )
    # Split the data set in X_train, y_train and X_test, y_test
    X_train, y_train, X_test, y_test = _split_target(dataset_with_feature)

    logger.info(f"Train's shape: {X_train.shape}")
    logger.info(f"Test's shape: {X_test.shape}")

    metrics_train, metrics_test = [], []
    best_model = []
    name_model = []
    for model, name in zip(models, name_models):

        logger.info(f"Training model...")
        fitted_model = _train_model(model, X_train, y_train)
        best_model.append(fitted_model)
        name_model.append(name)
        logger.info("Model trained.")

        lbl = LabelEncoder()
        
        train_metrics, test_metrics = (
             evaluate_model(fitted_model, 
                            X=features[0], 
                            y=features[1], 
                            classes=lbl.classes
                            )
             for features in ((X_train, y_train), (X_test, y_test))
        )
        metrics_test.append(test_metrics)
        metrics_train.append(train_metrics)

        metrics_msg = "=" * 50 + " Metrics " + "=" * 50
        logger.info(metrics_msg)
        logger.info(fitted_model)
        logger.info(f"Train: {train_metrics.get('overall')}")
        logger.info(f"Test: {test_metrics.get('overall')}")
        logger.info("=" *len(metrics_msg))

    # choix du meilleur model avec le f1_score
    f1_score = [res.get('overall').get('f1') for res in metrics_test]
    index_best_model = np.argmax(f1_score)
    
    return {
         'model': best_model[index_best_model],
         'metrics': {
             'name_model': name_model[index_best_model],
             'train': metrics_train[index_best_model].get('overall'),
             'test': metrics_test[index_best_model].get('overall')
         }
     }
