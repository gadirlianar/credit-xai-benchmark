import os
import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml
import numpy as np

def get_logger(name: str = "credit_xai", level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger with standard formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def set_seeds(seed: int = 42) -> None:
    """Sets deterministic random seeds and prevents OpenMP thread contention deadlocks on macOS."""
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def ensure_dirs(*dirs: Union[str, Path]) -> None:
    """Ensures that specified directories exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Saves a dictionary to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

def load_json(path: Union[str, Path]) -> Any:
    """Loads a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Saves data to a formatted JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
