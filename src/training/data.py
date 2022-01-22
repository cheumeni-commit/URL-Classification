import logging

import pandas as pd

from src.constants import (c_INDEX
                          )

from src.io import get_data

logger = logging.getLogger(__name__)


def _copy_dataSet(data: dict)->dict:
	return {'data_1': data.get('data_1').copy(),
            'data_2': data.get('data_2').copy(),
			'data_3': data.get('data_3').copy(),
			'data_4': data.get('data_4').copy(),
			'data_5': data.get('data_5').copy()
    	} 


def _get_concatenate_dataset(**kwargs)->pd.DataFrame:
	return (pd.concat(list(kwargs.values())).reset_index()).drop(c_INDEX, axis=1)

	
def build_dataset():
	"Build dataset with text transformation"

	logger.info("Copy of initial Dataset")
	catalog = get_data()
	# Copy of dataset
	logger.info("Extraction of Class and text Column")
	dataset_copy = _copy_dataSet(catalog)
	# dataset with text transaction and class
	dataset = _get_concatenate_dataset(**dataset_copy)
    
	return dataset