import os
import torch
import random
import numpy as np
from datetime import datetime
import logging


def set_determinism(seed: int=1, benchmark: bool=False, determinism: bool=True):
    assert sum([benchmark, determinism]) <= 1, "You can only set one of benchmark or determinism to True"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = benchmark

    if determinism:
        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    LOG_DEST = "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, LOG_DEST)),
            logging.StreamHandler()
        ])
