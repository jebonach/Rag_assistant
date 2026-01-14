from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_config(cfg: Any) -> Dict[str, Any]:
    d = {}
    for k, v in cfg.__dict__.items():
        d[k] = str(v) if isinstance(v, Path) else v
    for k in sorted(d):
        print(f"{k}: {d[k]}")
    return d
