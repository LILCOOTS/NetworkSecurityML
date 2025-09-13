import logging
import logging.config
import os
import sys
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H-%M-%S')}.log"

LOGS_PATH = os.path.join(os.getcwd(), LOG_DIR, LOG_FILE)
os.makedirs(LOGS_PATH, exist_ok=True)

LOGS_FILE_PATH = os.path.join(LOGS_PATH, LOG_FILE)

logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)