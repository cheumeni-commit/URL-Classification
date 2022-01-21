from collections import Counter
import joblib

import numpy as np

from src.cli import context
from src.constants import (c_SAVE_MODEL,
                           c_TEXT_TRANSFORMES,
                           c_DETAIL_ECRITURE
                          )
from src.training.features import build_predict_set


def decode(y, dictionary):
    classes = []
    for _, item in enumerate(y):
        classes.append(dictionary[str(item)])
    return classes


def _load_model():
    loaded_model = joblib.load(str(context.dirs.raw_store_dir) + '/' + c_SAVE_MODEL)
    return loaded_model


def _model_inference(model, X, lexiques, do_probabilities=False):

    X = predict_dataset(X, lexiques, c_DETAIL_ECRITURE)

    if do_probabilities:
        pred = model.predict_proba(X)
    else:
        pred = model.predict(X)
    return pred


def predict(X, lexiques, *, do_probabilities=False):
    model = _load_model()
    result = _model_inference(model, X, lexiques, do_probabilities)
    return result


def _encoding(corpus, lexiques):
    X_features = _encodage_text(corpus, lexiques)
    return X_features


def _encodage_text(corpus, lexique):

    Mat = np.zeros([len(corpus), len(lexique)], dtype=np.float64)

    for i, corp in enumerate(corpus):
        for j, lex in enumerate(lexique):
            counts_dict = {k:v for k,v in Counter(corp.split(" ")).most_common(5000)}
            if lex in counts_dict:
                Mat[i, j] = counts_dict.get(lex)

    return Mat
    

def predict_transaction(transaction):
    corpus = build_predict_set(transaction)
    X_features = _encoding(corpus)
    return X_features


def predict_dataset(dataset, lexiques, transactionColumn):
    dataset = build_predict_set(dataset, column_ecriture=transactionColumn)
    X = dataset.loc[:,c_TEXT_TRANSFORMES].values
    X_features = _encoding(X, lexiques)
    return X_features