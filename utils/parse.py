import os
from pathlib import Path
from typing import Any

import yaml


def parse_yaml(path: str) -> dict[str, Any]:
    """
    Parse yaml file and return dict.

    Parameters
    ----------
    path : str
        Path to yaml file.

    Returns
    -------
    dict[str, Any]
        Dict of yaml file.
    """
    if not os.path.exists(path):
        raise ValueError(f"File {path} does not exist.")
    with open(path, "r") as f:
        dct = yaml.safe_load(f.read())
        dct["filename"] = Path(path).stem
    return dct
