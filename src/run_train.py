"""
Model training script.
"""
import time
import logging

from src.cli import context
from src.io import save_training_output, load_data_train
from src.training.models import get_model
from src.train import train


logger = logging.getLogger(__name__)


def main():
    start = time.time()
    logger.info("Starting training job...")

    models, name_models, param_models = get_model()

    dataset = load_data_train()
    model_metrics = train(models, name_models, param_models, context, dataset)
    save_training_output(model_metrics, directory=context.dirs.raw_store_dir)
    run_duration = time.time() - start
    logger.info("Training job done...")
    logger.info(f"Took {run_duration} seconds to execute")


if __name__ == '__main__':
    main()