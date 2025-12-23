import os
from datetime import datetime
import sys
import logging

log_file_name = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.LOG"

log_path = os.path.join(os.getcwd(), 'logs')

os.makedirs(log_path, exist_ok = True)
    
final_log_file_path = os.path.join(log_path, log_file_name)

logging.basicConfig(
    filename= final_log_file_path,
    format = '[%(asctime)s] %(lineno)d -%(levelname)s -%(message)s',
    level=logging.INFO
)