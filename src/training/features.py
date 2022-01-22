import re

import pandas as pd

from src.constants import (c_HTML, 
						   c_HTTPS,
						   c_HTTP, 
						   c_COM, 
						   c_WWW,
						   c_FR,
						   c_TEXT_TRANSFORMES,
						   c_MIN_LABEL_COUNT,
						   c_TAGS,
						   c_OCCURRENCE,
						   c_DICO
						   )
from src.cli import context
from src.io import save_lexique


def _preprocess_test(data, column_ecriture):
	"""Conditional preprocessing on our text"""
	
	transaction = []
	if column_ecriture is not None:
		for text in data[column_ecriture]:
			text = _preprocess(text)
			transaction.append(text)
		data[c_TEXT_TRANSFORMES] = transaction
	else: 
		data[c_TEXT_TRANSFORMES] = _preprocess(data)
	return data


def _preprocess(text):
	# Lower
	text = text.lower()
	# Spacing and filters
	text = re.sub('[^a-zA-Z]', ' ', text)
	text = text.split()
	text = [word.lower() for word in text if len(word) > 1]
	text = ' '.join(text)
	text = _replace_text(text, [c_FR, c_HTML, c_COM, c_HTTPS, c_WWW, c_HTTP])
	return text


def _replace_text(text, catalogue):
	for cat in catalogue:
		text = text.replace(cat, '')
	return text


def _get_tags(data:pd.DataFrame, dico:dict)->pd.DataFrame:
	data[c_TAGS] = (data.target).apply(lambda raw:dico.get(raw.split('"')[0])[0])
	return data


def _get_occurence(data:pd.DataFrame, dico:dict)->pd.DataFrame:
	data[c_OCCURRENCE] = (data.target).apply(lambda raw:dico.get(raw.split('"')[0])[1])
	return data


def _get_dico_tags_class(data:pd.DataFrame)->dict:
	if isinstance(data, pd.DataFrame):
		dico = {k: (f'tag_{i}', v) for i, (k,v) in enumerate((data.target).value_counts().items())}
		dico_save = {f'tag_{i}': k for i, (k,v) in enumerate((data.target).value_counts().items())}
		save_lexique(dico_save, path=context.dirs.config / c_DICO)
	return dico


def build_train_test_set(data, *, column_ecriture=None):

	# building of dictionnary between tags and class
	dico = _get_dico_tags_class(data)
	# add tags in dataset
	dataset = _get_tags(data, dico)
    # add occurrence in dataset
	dataset = _get_occurence(dataset, dico)
	# filter dataset
	df_filter = dataset[dataset.occurrences>=c_MIN_LABEL_COUNT]
	dataset = df_filter.reset_index().drop('index', axis=1)
    # add column text transform
	dataset = _preprocess_test(dataset, column_ecriture)
	
	return dataset


def build_predict_set(data, *, column_ecriture=None):
	# add column text transform
	dataset = _preprocess_test(data, column_ecriture)
	return dataset

