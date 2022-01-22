import logging

from src.cli import context
from src.run_dataset import main as main_datatset
from src.run_train import main as main_train
from src.run_predict import main as main_predict


logger = logging.getLogger(__name__)


def _choice_env():
    if context.command == 'run_dataset':
        main_datatset()
    elif context.command == 'run_train':
        main_train()
    elif context.command == 'run_predict':
        if context.input_data is None:
            main_predict(None)
        else:
            main_predict(context.input_data)


if __name__ == '__main__':
    logger.debug("main of project")

    if context.environment == 'dev':
        _choice_env()
    elif context.environment == 'prod':
        _choice_env()