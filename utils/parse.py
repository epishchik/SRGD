import os
from pathlib import Path
from typing import Any

import yaml


def parse_yaml(path: str) -> dict[str, Any]:
    """
        Функция парсинга конфигурационного yaml файла в python dict.

        Parameters
        ----------
        path : str
            Путь к конфигурационному файлу.

        Returns
        -------
        dict[str, Any]
            Словарь параметр - значение параметра.
    """
    if not os.path.exists(path):
        raise ValueError(f'File {path} does not exist.')
    with open(path, 'r') as f:
        dct = yaml.safe_load(f.read())
        dct['filename'] = Path(path).stem
    return dct
