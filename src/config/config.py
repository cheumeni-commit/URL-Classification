
from dataclasses import dataclass
import logging
import yaml

from src.constants import (c_DEV,
                           c_PROD 
                           )


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    model: dict


def load_param(context, file_name):
    read_data = []
    with open(str(context.dirs.config) + "/" + file_name, 'r') as fp:
        read_data.append(yaml.safe_load(fp))
    return read_data


def load_config_file(context):
    try:
        if context.environment == 'prod':
            read_data = load_param(context, c_PROD)
        else:
            read_data = load_param(context, c_DEV)
    except:
            logger.info("yml file don't find inside directories")
    return read_data


def get_config(context) -> Config:
    config = load_config_file(context)
    return Config(*config)