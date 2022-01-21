import logging

from src.cli import context
from src.constants import c_INTERMEDIATE
from src.io import save_dataset
from src.training.data import build_dataset


logger = logging.getLogger(__name__)

def main():
    "Generate dataset and store it on disk as a csv file"
    logger.info("Building dataset...")
    dataset = build_dataset()
    # save dataset
    save_dataset(dataset, path= context.dirs.intermediate / c_INTERMEDIATE)


if __name__ == '__main__':
    main()