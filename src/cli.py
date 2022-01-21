from datetime import datetime
import argparse

from src.config.config import get_config
from src.contexts import context


parser = argparse.ArgumentParser()

parser.add_argument('command', choices=['run_dataset', 'run_train', 'run_predict'])
parser.add_argument("-i", "--input-data")
parser.add_argument('--environment', choices=['dev', 'production'])
parser.add_argument('--show-time', action='store_true')

args = parser.parse_args()

for k, v in vars(args).items():
    setattr(context, k, v)
    
context.config = get_config(context)
