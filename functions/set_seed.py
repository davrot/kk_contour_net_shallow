import torch
import numpy as np


def set_seed(seed: int, logger) -> None:
    # set seed for all used modules
    logger.info(f"set seed to {seed}")
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed=seed)
    np.random.seed(seed=seed)
