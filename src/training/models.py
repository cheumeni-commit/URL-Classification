from collections import defaultdict
import logging
from typing import List

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.cli import context

_MODELS_REGISTRY_ = {'RandomForestClassifier': RandomForestClassifier,
                     'xgb_Classifier': xgb.XGBClassifier,
                     'SGDClassifier': SGDClassifier
                     }

logger = logging.getLogger(__name__)


def _loadModels()-> List:
    model = []
    name_model = []
    param_model = []
    for _, v in context.config.model.items():
        model.append(_MODELS_REGISTRY_[v.get('name')])
        name_model.append(v.get('name'))
        param_model.append(v.get('params'))
    return model, name_model, param_model


def get_model():
    """ Load Model """
    try:
        model, name_model, param_model = _loadModels()
    except:
        logger.info("The model is not available ")

    return model, name_model, param_model