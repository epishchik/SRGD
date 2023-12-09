from pathlib import Path
from typing import Any

import yaml


def parse_yaml(path: str) -> dict[str, Any]:
    with open(path, 'r') as f:
        dct = yaml.safe_load(f.read())
        dct['filename'] = Path(path).stem
    return dct