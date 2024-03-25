import logging

import numpy as np
import requests
from colorlog import ColoredFormatter
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from Config import Config


def configure_logger():
    logger = logging.getLogger()
    logger.setLevel(Config.LOG_LEVEL)
    logger.handlers.clear()

    # # For Log file
    file_handler = logging.FileHandler('MSD_AI.log', mode='w')
    file_handler.setFormatter(logging.Formatter("s%(asctime)s - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s"))

    # For Console
    console_format = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(lineno)d - %(levelname)s - [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'blue,bg_white'
        },
        secondary_log_colors={},
        style='%'
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_format)

    if Config.FILE_LOGGING:
        logger.addHandler(file_handler)
    if Config.DEBUG_LOGGING_CONSOLE:
        logger.addHandler(console_handler)


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def log_scale(value, epsilon=1e-3):
    if value == 0:
        return -1
    return np.log(value+epsilon)


def requests_retry_session(
    retries=15, backoff_factor=1.5, status_forcelist=(500, 502, 504), session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session