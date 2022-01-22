"""
Prediction pipeline
"""
import time
import logging

import pandas as pd
import numpy as np

from src.contexts import context
from src.constants import (c_PREDICTIONS_DATA,
                           c_LEXIQUE,
                           c_PREDICTIONS,
                           c_LABELS,
                           c_DICO
                          )
from src.predict.main import predict, decode
from src.io import load_json_file, save_prediction


logger = logging.getLogger(__name__)


def prediction_transform(prob, mapping, dico, X):

    y_pred = proba_max(prob)

    predictions = [{'url':X[i], 'class':dico.get(k),'score':float(v)} \
                    for k,v,i in zip(decode(y_pred, mapping), 
                    (pb[y_] for pb,y_ in zip(prob, y_pred)), range(len(X)))]
    return predictions


def proba_max(prob):
    return np.argmax(prob, axis=1)
    

def load_data(args):
    if args != None :
        X = load_json_file(args)
    else:
        X = load_json_file(context.dirs.test_dir / c_PREDICTIONS_DATA)
    return X


def main(args):
    start = time.time()
    logger.info("Starting prediction job...")

    # load data 
    X = load_data(args)
    # load lexique
    lexiques = load_json_file(context.dirs.config / c_LEXIQUE)
    labels = load_json_file(context.dirs.config / c_LABELS)
    dico = load_json_file(context.dirs.config / c_DICO)
    # transform data in DataFrame
    X = pd.DataFrame(X)
    # predict
    prob = predict(X, lexiques, do_probabilities=True)
    predictions = prediction_transform(prob, labels, dico, X.url)
    
    # save predictions
    save_prediction(predictions, path=context.dirs.raw_store_dir /c_PREDICTIONS)

    run_duration = time.time() - start
    logger.info("Prediction job done.")
    logger.info(f"Prediction took {run_duration:.2f}s to execute")
    return None


if __name__ == '__main__':
    main()