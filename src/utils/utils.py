import logging
import random

import numpy as np
import torch


def setup_logger() -> logging.Logger:
    """Sets logger configuration

    Returns:
        logging.Logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers if they exist
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)

    # Create a simple formatter and add it to the handler
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
